import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from collections import deque
from enum import Enum

class PersonState(Enum):
    STANDING = "Standing"
    SITTING = "Sitting"
    FALLING = "Falling"
    FALLEN = "Fallen"
    UNKNOWN = "Unknown"

class FallDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.model = None
        self.movenet = None
        self.input_size = 192
        
        # Detection parameters
        self.fall_vel_threshold = 0.7
        self.torso_angle_thresh_deg = 45
        self.post_fall_inactivity_thresh = 0.02
        self.post_fall_check_seconds = 1.0
        self.keypoint_score_thresh = 0.2
        
        # Buffer for temporal analysis
        self.buffer_seconds = 2.0
        self.buffer_len = int(self.buffer_seconds * fps)
        self.feat_buffer = deque(maxlen=self.buffer_len)
        
        # Previous frame data
        self.prev_keypoints = None
        self.prev_timestamp = None
        
        # Keypoint indices (COCO format)
        self.shoulder_left = 5
        self.shoulder_right = 6
        self.hip_left = 11
        self.hip_right = 12
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the MoveNet model from TensorFlow Hub"""
        try:
            print("Loading MoveNet model...")
            self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.movenet = self.model.signatures['serving_default']
            print("MoveNet model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simple detection method
            self.model = None
            self.movenet = None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for MoveNet input"""
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img.astype(np.int32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def run_inference(self, frame):
        """Run pose estimation inference on frame"""
        if self.movenet is None:
            # Return dummy keypoints if model failed to load
            return np.zeros((17, 3))
        
        try:
            inp = self.preprocess_frame(frame)
            outputs = self.movenet(tf.constant(inp, dtype=tf.int32))
            keypoints = outputs['output_0'].numpy()  # shape (1, 1, 17, 3)
            return keypoints[0, 0, :, :]  # Return 17x3 array
        except Exception as e:
            print(f"Inference error: {e}")
            return np.zeros((17, 3))
    
    def compute_midpoint(self, a, b):
        """Compute midpoint between two points"""
        return (a + b) / 2.0
    
    def compute_torso_angle_deg(self, keypoints):
        """Compute torso angle in degrees"""
        try:
            # Get shoulder and hip midpoints
            sh = self.compute_midpoint(
                keypoints[self.shoulder_left, :2], 
                keypoints[self.shoulder_right, :2]
            )
            hp = self.compute_midpoint(
                keypoints[self.hip_left, :2], 
                keypoints[self.hip_right, :2]
            )
            
            # Calculate angle
            dy = hp[0] - sh[0]  # positive = hip below shoulder
            dx = hp[1] - sh[1]
            angle_rad = np.arctan2(abs(dx), abs(dy) + 1e-9)
            return np.degrees(angle_rad)
        except:
            return 0.0
    
    def mid_hip_y_normalized(self, keypoints):
        """Get normalized mid-hip Y coordinate"""
        try:
            vals = []
            if keypoints[self.hip_left, 2] > self.keypoint_score_thresh:
                vals.append(keypoints[self.hip_left, 0])
            if keypoints[self.hip_right, 2] > self.keypoint_score_thresh:
                vals.append(keypoints[self.hip_right, 0])
            
            if len(vals) == 0:
                return None
            return float(np.mean(vals))
        except:
            return None
    
    def compute_activity_energy_norm(self, keypoints_prev, keypoints_curr, frame_h, frame_w):
        """Compute normalized activity energy"""
        if keypoints_prev is None:
            return 0.0
        
        try:
            # Calculate displacement
            disp = np.linalg.norm(
                ((keypoints_curr[:, :2] - keypoints_prev[:, :2]) * np.array([frame_h, frame_w])), 
                axis=1
            )
            
            # Only consider valid keypoints
            valid = (keypoints_curr[:, 2] > self.keypoint_score_thresh) & \
                   (keypoints_prev[:, 2] > self.keypoint_score_thresh)
            
            if not valid.any():
                return 0.0
            
            energy = np.sum(disp[valid])
            diag = np.sqrt(frame_h**2 + frame_w**2)
            return float(energy / (diag + 1e-9))
        except:
            return 0.0
    
    def process_frame(self, frame, keypoints, timestamp):
        """Process frame and return fall detection results"""
        h, w = frame.shape[:2]
        
        # Compute features
        mid_hip = self.mid_hip_y_normalized(keypoints)
        torso_angle = self.compute_torso_angle_deg(keypoints)
        activity = self.compute_activity_energy_norm(self.prev_keypoints, keypoints, h, w)
        
        # Add to buffer
        self.feat_buffer.append((timestamp, mid_hip, torso_angle, activity))
        
        # Compute hip velocity
        hip_vel = None
        if len(self.feat_buffer) >= 2 and mid_hip is not None:
            newest_ts, newest_hip, _, _ = self.feat_buffer[-1]
            
            # Find previous sample for velocity calculation
            prev_sample = None
            for sample in reversed(list(self.feat_buffer)[:-1]):
                if sample[0] <= newest_ts - 0.12:  # At least 120ms ago
                    prev_sample = sample
                    break
            
            if prev_sample is None and len(self.feat_buffer) >= 2:
                prev_sample = self.feat_buffer[-2]
            
            if prev_sample and prev_sample[1] is not None:
                prev_ts, prev_hip, _, _ = prev_sample
                dt = newest_ts - prev_ts
                if dt > 0:
                    hip_vel = (newest_hip - prev_hip) / dt
        
        # Fall detection logic
        state = PersonState.STANDING
        fall_confidence = 0.0
        
        if hip_vel is not None:
            # Check velocity threshold
            vel_check = hip_vel > self.fall_vel_threshold
            torso_check = torso_angle >= self.torso_angle_thresh_deg
            
            # Check post-fall inactivity
            post_fall_ok = False
            if vel_check and torso_check:
                tail = [f for f in self.feat_buffer if f[0] >= timestamp - self.post_fall_check_seconds]
                if len(tail) > 0:
                    avg_activity = np.mean([f[3] for f in tail])
                    post_fall_ok = avg_activity < self.post_fall_inactivity_thresh
            
            # Determine state and confidence
            if vel_check and torso_check and post_fall_ok:
                state = PersonState.FALLEN
                fall_confidence = min(1.0, (hip_vel / self.fall_vel_threshold) * 0.8 + 0.2)
            elif vel_check or (torso_check and torso_angle > 60):
                state = PersonState.FALLING
                fall_confidence = min(0.8, (hip_vel / self.fall_vel_threshold) * 0.6 + 0.1)
            elif torso_angle > 30:
                state = PersonState.SITTING
                fall_confidence = 0.1
            else:
                state = PersonState.STANDING
                fall_confidence = 0.0
        
        # Update previous frame data
        self.prev_keypoints = keypoints.copy() if keypoints is not None else None
        self.prev_timestamp = timestamp
        
        return state, fall_confidence
    
    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton on frame"""
        if keypoints is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # COCO pose connections
        connections = [
            (5, 6),   # shoulders
            (5, 7),   # left shoulder to elbow
            (7, 9),   # left elbow to wrist
            (6, 8),   # right shoulder to elbow
            (8, 10),  # right elbow to wrist
            (5, 11),  # left shoulder to hip
            (6, 12),  # right shoulder to hip
            (11, 12), # hips
            (11, 13), # left hip to knee
            (13, 15), # left knee to ankle
            (12, 14), # right hip to knee
            (14, 16), # right knee to ankle
        ]
        
        # Draw keypoints
        for i, (y, x, score) in enumerate(keypoints):
            if score > self.keypoint_score_thresh:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        
        # Draw connections
        for start_idx, end_idx in connections:
            if (keypoints[start_idx, 2] > self.keypoint_score_thresh and 
                keypoints[end_idx, 2] > self.keypoint_score_thresh):
                
                start_point = (
                    int(keypoints[start_idx, 1] * w),
                    int(keypoints[start_idx, 0] * h)
                )
                end_point = (
                    int(keypoints[end_idx, 1] * w),
                    int(keypoints[end_idx, 0] * h)
                )
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        return frame