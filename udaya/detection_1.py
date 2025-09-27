import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from collections import deque
from enum import Enum
import threading
import winsound  # For Windows alert sound (replace with appropriate for your OS)

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

INPUT_SIZE = 192

# Keypoint indices for MoveNet
class KeypointIndex:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

class PersonState(Enum):
    STANDING = "Standing"
    SITTING = "Sitting"
    LYING = "Lying"
    FALLING = "Falling"
    FALLEN = "Fallen"
    UNKNOWN = "Unknown"

class FallDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.buffer_seconds = 2.0
        self.buffer_len = int(self.buffer_seconds * fps)
        
        # Detection thresholds (tuned for elderly movement patterns)
        self.keypoint_confidence_thresh = 0.15  # Lower for better tracking
        self.fall_velocity_thresh = 0.35  # m/s normalized (elderly fall slower)
        self.torso_angle_lying_thresh = 60  # degrees from vertical
        self.torso_angle_falling_thresh = 45  # degrees from vertical
        self.height_drop_thresh = 0.25  # normalized height drop
        self.inactivity_thresh = 0.015  # post-fall movement threshold
        self.recovery_time_thresh = 3.0  # seconds to wait for recovery
        
        # Buffers for temporal analysis
        self.pose_buffer = deque(maxlen=self.buffer_len)
        self.state_buffer = deque(maxlen=10)
        self.fall_confidence_buffer = deque(maxlen=5)
        
        # State tracking
        self.current_state = PersonState.UNKNOWN
        self.fall_detected_time = None
        self.alert_triggered = False
        self.consecutive_fall_frames = 0
        self.baseline_height = None
        self.calibration_frames = 0
        
    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.int32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def run_inference(self, frame):
        inp = self.preprocess_frame(frame)
        outputs = movenet(tf.constant(inp, dtype=tf.int32))
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        return keypoints
    
    def compute_center_of_mass(self, kpts):
        """Compute approximate center of mass from keypoints"""
        # Weight different body parts for COM calculation
        weights = {
            'head': [(KeypointIndex.NOSE, 0.08)],
            'torso': [(KeypointIndex.LEFT_SHOULDER, 0.15), (KeypointIndex.RIGHT_SHOULDER, 0.15),
                     (KeypointIndex.LEFT_HIP, 0.15), (KeypointIndex.RIGHT_HIP, 0.15)],
            'limbs': [(KeypointIndex.LEFT_KNEE, 0.08), (KeypointIndex.RIGHT_KNEE, 0.08),
                     (KeypointIndex.LEFT_ANKLE, 0.08), (KeypointIndex.RIGHT_ANKLE, 0.08)]
        }
        
        total_weight = 0
        com_y = 0
        com_x = 0
        
        for category, points in weights.items():
            for idx, w in points:
                if kpts[idx, 2] > self.keypoint_confidence_thresh:
                    com_y += kpts[idx, 0] * w
                    com_x += kpts[idx, 1] * w
                    total_weight += w
        
        if total_weight > 0:
            return com_y / total_weight, com_x / total_weight
        return None, None
    
    def compute_body_orientation(self, kpts):
        """Calculate body orientation angles"""
        # Torso angle (shoulder to hip line)
        shoulder_center = self._get_midpoint(kpts, KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER)
        hip_center = self._get_midpoint(kpts, KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP)
        
        if shoulder_center and hip_center:
            dy = hip_center[0] - shoulder_center[0]
            dx = hip_center[1] - shoulder_center[1]
            torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9))
        else:
            torso_angle = None
        
        # Leg angle (hip to ankle)
        leg_angle = None
        if kpts[KeypointIndex.LEFT_HIP, 2] > self.keypoint_confidence_thresh and \
           kpts[KeypointIndex.LEFT_ANKLE, 2] > self.keypoint_confidence_thresh:
            dy = kpts[KeypointIndex.LEFT_ANKLE, 0] - kpts[KeypointIndex.LEFT_HIP, 0]
            dx = kpts[KeypointIndex.LEFT_ANKLE, 1] - kpts[KeypointIndex.LEFT_HIP, 1]
            leg_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9))
        
        return torso_angle, leg_angle
    
    def _get_midpoint(self, kpts, idx1, idx2):
        """Get midpoint between two keypoints"""
        if kpts[idx1, 2] > self.keypoint_confidence_thresh and \
           kpts[idx2, 2] > self.keypoint_confidence_thresh:
            return ((kpts[idx1, 0] + kpts[idx2, 0]) / 2,
                   (kpts[idx1, 1] + kpts[idx2, 1]) / 2)
        return None
    
    def compute_motion_energy(self, kpts_prev, kpts_curr):
        """Calculate overall motion energy between frames"""
        if kpts_prev is None:
            return 0.0
        
        motion = 0.0
        valid_points = 0
        
        for i in range(len(kpts_curr)):
            if kpts_curr[i, 2] > self.keypoint_confidence_thresh and \
               kpts_prev[i, 2] > self.keypoint_confidence_thresh:
                displacement = np.linalg.norm(kpts_curr[i, :2] - kpts_prev[i, :2])
                motion += displacement
                valid_points += 1
        
        return motion / (valid_points + 1e-9)
    
    def detect_fall_features(self, current_features, buffer):
        """Analyze multiple fall indicators"""
        if len(buffer) < 10:
            return 0.0  # Not enough data
        
        fall_score = 0.0
        recent_data = list(buffer)[-30:]  # Last second of data (assuming 30fps)
        
        # 1. Rapid height drop
        heights = [f['com_y'] for f in recent_data if f['com_y'] is not None]
        if len(heights) >= 2:
            height_drop = max(heights[:5]) - min(heights[-5:])  # Compare early vs late
            if height_drop > self.height_drop_thresh:
                fall_score += 0.3
        
        # 2. Velocity spike
        if current_features['velocity_y'] is not None:
            if current_features['velocity_y'] > self.fall_velocity_thresh:
                fall_score += 0.25
        
        # 3. Torso becomes horizontal
        if current_features['torso_angle'] is not None:
            if current_features['torso_angle'] > self.torso_angle_falling_thresh:
                fall_score += 0.25
        
        # 4. Sudden loss of vertical posture
        angles = [f['torso_angle'] for f in recent_data if f['torso_angle'] is not None]
        if len(angles) >= 2:
            angle_change = angles[-1] - np.mean(angles[:5]) if len(angles) >= 5 else 0
            if angle_change > 30:  # Sudden angle change
                fall_score += 0.2
        
        # 5. Post-fall inactivity
        recent_motion = [f['motion_energy'] for f in recent_data[-10:] if f['motion_energy'] is not None]
        if recent_motion and np.mean(recent_motion) < self.inactivity_thresh:
            fall_score += 0.2
        
        return min(fall_score, 1.0)
    
    def determine_person_state(self, features, fall_confidence):
        """Determine person's current state based on features"""
        if features['torso_angle'] is None:
            return PersonState.UNKNOWN
        
        # High fall confidence
        if fall_confidence > 0.6:
            if self.fall_detected_time is None:
                self.fall_detected_time = time.time()
                return PersonState.FALLING
            elif time.time() - self.fall_detected_time > self.recovery_time_thresh:
                return PersonState.FALLEN
            else:
                return PersonState.FALLING
        
        # Check if lying down
        if features['torso_angle'] > self.torso_angle_lying_thresh:
            # Check if this is intentional lying or result of fall
            if self.fall_detected_time and time.time() - self.fall_detected_time < 10:
                return PersonState.FALLEN
            return PersonState.LYING
        
        # Check if sitting (medium height, moderate angle)
        if features['com_y'] is not None and self.baseline_height is not None:
            height_ratio = features['com_y'] / (self.baseline_height + 1e-9)
            if 0.4 < height_ratio < 0.7 and features['torso_angle'] < 30:
                return PersonState.SITTING
        
        # Default to standing if upright
        if features['torso_angle'] < 30:
            return PersonState.STANDING
        
        return PersonState.UNKNOWN
    
    def calibrate_baseline(self, features):
        """Calibrate baseline measurements during initial frames"""
        if self.calibration_frames < 30:  # Calibrate for 1 second
            if features['com_y'] is not None:
                if self.baseline_height is None:
                    self.baseline_height = features['com_y']
                else:
                    # Running average
                    self.baseline_height = 0.9 * self.baseline_height + 0.1 * features['com_y']
            self.calibration_frames += 1
    
    def process_frame(self, frame, keypoints, timestamp):
        """Main processing pipeline for fall detection"""
        h, w = frame.shape[:2]
        
        # Extract features
        com_y, com_x = self.compute_center_of_mass(keypoints)
        torso_angle, leg_angle = self.compute_body_orientation(keypoints)
        
        # Calculate velocity if we have previous data
        velocity_y = None
        if len(self.pose_buffer) >= 5:
            prev_data = self.pose_buffer[-5]
            if com_y is not None and prev_data['com_y'] is not None:
                dt = timestamp - prev_data['timestamp']
                if dt > 0:
                    velocity_y = (com_y - prev_data['com_y']) / dt
        
        # Calculate motion energy
        prev_kpts = self.pose_buffer[-1]['keypoints'] if self.pose_buffer else None
        motion_energy = self.compute_motion_energy(prev_kpts, keypoints)
        
        # Store current features
        current_features = {
            'timestamp': timestamp,
            'keypoints': keypoints.copy(),
            'com_y': com_y,
            'com_x': com_x,
            'torso_angle': torso_angle,
            'leg_angle': leg_angle,
            'velocity_y': velocity_y,
            'motion_energy': motion_energy
        }
        
        self.pose_buffer.append(current_features)
        
        # Calibrate baseline during initial frames
        self.calibrate_baseline(current_features)
        
        # Detect fall indicators
        fall_confidence = self.detect_fall_features(current_features, self.pose_buffer)
        self.fall_confidence_buffer.append(fall_confidence)
        
        # Smooth fall confidence over multiple frames
        avg_fall_confidence = np.mean(self.fall_confidence_buffer)
        
        # Determine person state
        new_state = self.determine_person_state(current_features, avg_fall_confidence)
        
        # State transition logic
        if new_state != self.current_state:
            # Check for critical transitions
            if new_state in [PersonState.FALLING, PersonState.FALLEN]:
                self.consecutive_fall_frames += 1
                if self.consecutive_fall_frames >= 3:  # Confirm over multiple frames
                    self.current_state = new_state
                    if not self.alert_triggered:
                        self.trigger_alert(frame)
            else:
                self.consecutive_fall_frames = 0
                if self.current_state in [PersonState.FALLING, PersonState.FALLEN] and \
                   new_state in [PersonState.STANDING, PersonState.SITTING]:
                    # Person recovered
                    self.fall_detected_time = None
                    self.alert_triggered = False
                self.current_state = new_state
        
        return self.current_state, avg_fall_confidence
    
    def trigger_alert(self, frame):
        """Trigger alert when fall is detected"""
        self.alert_triggered = True
        print("\n🚨 FALL DETECTED! 🚨")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save snapshot
        filename = f"fall_detected_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        
        # Play alert sound (Windows)
        try:
            winsound.Beep(1000, 500)  # Frequency, Duration
        except:
            pass  # Sound might not work on all systems
        
        # Here you could add:
        # - Send SMS/Email notification
        # - Trigger IoT alarm
        # - Call emergency contacts
        # - Send to monitoring station

def draw_skeleton(frame, keypoints, w, h, confidence_thresh=0.2):
    """Draw pose skeleton on frame"""
    # Define connections
    connections = [
        (KeypointIndex.LEFT_EAR, KeypointIndex.LEFT_EYE),
        (KeypointIndex.LEFT_EYE, KeypointIndex.NOSE),
        (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
        (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
        (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
        (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
        (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
        (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
        (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
        (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
        (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
        (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
        (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
        (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
        (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
        (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE)
    ]
    
    # Draw connections
    for connection in connections:
        kpt1, kpt2 = connection
        if keypoints[kpt1, 2] > confidence_thresh and keypoints[kpt2, 2] > confidence_thresh:
            pt1 = (int(keypoints[kpt1, 1] * w), int(keypoints[kpt1, 0] * h))
            pt2 = (int(keypoints[kpt2, 1] * w), int(keypoints[kpt2, 0] * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] > confidence_thresh:
            cx, cy = int(kp[1] * w), int(kp[0] * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Get camera FPS (or set manually)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60:
        fps = 30  # Default to 30 FPS
    
    # Initialize fall detector
    detector = FallDetector(fps=fps)
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    display_fps = 0
    
    print("Fall Detection System for Elderly Care - Started")
    print("Press 'q' to quit, 's' to save snapshot")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        timestamp = time.time()
        
        # Run pose detection
        keypoints = detector.run_inference(frame)
        
        # Process frame for fall detection
        state, fall_confidence = detector.process_frame(frame, keypoints, timestamp)
        
        # Draw skeleton
        draw_skeleton(frame, keypoints, w, h)
        
        # Create status panel
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Display status
        status_color = (0, 255, 0)  # Green by default
        if state == PersonState.FALLING:
            status_color = (0, 165, 255)  # Orange
        elif state == PersonState.FALLEN:
            status_color = (0, 0, 255)  # Red
        elif state == PersonState.UNKNOWN:
            status_color = (128, 128, 128)  # Gray
        
        # Status text
        cv2.putText(frame, f"Status: {state.value}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Fall confidence bar
        bar_width = int(200 * fall_confidence)
        bar_color = (0, 255, 0) if fall_confidence < 0.3 else \
                   (0, 165, 255) if fall_confidence < 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (10, 50), (210, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), bar_color, -1)
        cv2.putText(frame, f"Fall Risk: {fall_confidence:.1%}", (220, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Additional info
        info_text = f"FPS: {display_fps:.1f}"
        cv2.putText(frame, info_text, (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert overlay for falls
        if state in [PersonState.FALLING, PersonState.FALLEN]:
            alert_overlay = frame.copy()
            cv2.rectangle(alert_overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 255), 3)
            cv2.putText(alert_overlay, "FALL DETECTED!", (w//4 + 20, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(alert_overlay, "Alerting emergency contacts...", (w//4 + 20, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame = cv2.addWeighted(alert_overlay, 0.7, frame, 0.3, 0)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            elapsed = time.time() - fps_start_time
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display frame
        cv2.imshow("Elderly Fall Detection System", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem stopped.")

if __name__ == "__main__":
    main()