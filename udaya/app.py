from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import base64
import numpy as np
import json
from datetime import datetime, timedelta
from detection_3 import FallDetector
from enum import Enum
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class PersonState(Enum):
    STANDING = "Standing"
    SITTING = "Sitting"
    FALLING = "Falling"
    FALLEN = "Fallen"
    UNKNOWN = "Unknown"

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FallAlert:
    id: str
    timestamp: datetime
    confidence: float
    state: PersonState
    level: AlertLevel
    resolved: bool = False
    emergency_contacted: bool = False

@dataclass
class SystemStatus:
    active: bool
    fps: float
    current_state: PersonState
    fall_confidence: float
    uptime: float
    total_alerts: int
    camera_status: str

class FallDetectionManager:
    def __init__(self):
        self.detector = None
        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_state = PersonState.UNKNOWN
        self.fall_confidence = 0.0
        self.fps = 0
        self.start_time = None
        self.alerts: List[FallAlert] = []
        self.alert_callbacks = []
        self.detection_thread = None
        self.frame_lock = threading.Lock()
        self.video_save_callbacks = []  # Add this line
        self.recorded_incidents = [] 
        
        # Settings
        self.sensitivity = 0.6
        self.alert_delay = 3  # seconds
        self.last_alert_time = 0
        self.last_status_emit_time = 0.0
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera and fall detector"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps > 60:
                fps = 30
                
            # Initialize fall detector
            self.detector = FallDetector(fps=fps)
            logger.info(f"Camera initialized successfully with FPS: {fps}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def draw_skeleton(self, frame, keypoints, w, h):
        """Draw keypoints using MoveNet output format (y, x, score)."""
        if keypoints is None:
            return frame

        try:
            if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 2:
                for yx_score in keypoints:
                    # keypoint format: [y, x, score]
                    if len(yx_score) >= 3 and float(yx_score[2]) < 0.2:
                        continue
                    y_norm = float(yx_score[0])
                    x_norm = float(yx_score[1])
                    x = int(x_norm * w)
                    y = int(y_norm * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        except Exception as e:
            logger.warning(f"Error drawing skeleton: {str(e)}")

        return frame

    def _map_external_state(self, external_state):
        """Map detector-specific state enums/strings to this module's PersonState."""
        try:
            if isinstance(external_state, Enum):
                state_str = external_state.value
            elif isinstance(external_state, str):
                state_str = external_state
            else:
                state_str = str(external_state)
        except Exception:
            state_str = "Unknown"

        mapping = {
            "Standing": PersonState.STANDING,
            "Sitting": PersonState.SITTING,
            "Falling": PersonState.FALLING,
            "Fallen": PersonState.FALLEN,
            "Lying": PersonState.FALLEN,
            "Unknown": PersonState.UNKNOWN,
        }
        return mapping.get(state_str, PersonState.UNKNOWN)
    
    def process_frame(self, frame):
        """Process a single frame for fall detection"""
        h, w = frame.shape[:2]
        timestamp = time.time()
        
        try:
            # Run pose detection
            keypoints = self.detector.run_inference(frame)
            
            # Process frame for fall detection
            state_external, fall_confidence = self.detector.process_frame(frame, keypoints, timestamp)
            state = self._map_external_state(state_external)
            
            # Update current state
            previous_state = self.current_state
            self.current_state = state
            self.fall_confidence = fall_confidence
            
            # Draw skeleton
            frame = self.draw_skeleton(frame, keypoints, w, h)
            
            # Draw status overlay
            frame = self.draw_status_overlay(frame, state, fall_confidence, w, h)
            
            # Check for fall alerts and handle video recording
            self.check_fall_alert(state, fall_confidence, timestamp, previous_state)
            
            # Check if video recording is complete
            if hasattr(self.detector, 'video_buffer'):
                if self.detector.alert_triggered and not self.detector.incident_video_saved:
                    if self.detector.video_buffer.is_recording_complete():
                        self.save_incident_video()
            
            return frame, state, fall_confidence
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, PersonState.UNKNOWN, 0.0
        
    def save_incident_video(self):
        """Save the complete incident video when recording is done"""
        try:
            timestamp = int(time.time())
            filename = f"fall_incident_{timestamp}.mp4"
            filepath = self.UPLOAD_FOLDER / filename
            
            # Save the video
            if self.detector.video_buffer.save_complete_incident_video(str(filepath)):
                self.detector.incident_video_saved = True
                
                # Store incident record
                incident = {
                    'id': f"incident_{timestamp}",
                    'timestamp': datetime.now().isoformat(),
                    'video_path': str(filepath),
                    'video_filename': filename,
                    'duration': 40,  # 20s before + 20s after
                    'alert_id': self.alerts[-1].id if self.alerts else None
                }
                self.recorded_incidents.append(incident)
                
                # Emit video saved event
                if 'socketio' in globals() and socketio is not None:
                    socketio.emit('video_saved', incident)
                
                # Trigger video save callbacks
                for callback in self.video_save_callbacks:
                    try:
                        callback(incident)
                    except Exception as e:
                        logger.error(f"Error in video save callback: {str(e)}")
                
                logger.info(f"✅ Incident video saved: {filepath}")
                return True
            else:
                logger.error("Failed to save incident video")
                return False
                
        except Exception as e:
            logger.error(f"Error saving incident video: {str(e)}")
            return False
    
    def draw_status_overlay(self, frame, state, fall_confidence, w, h):
        """Draw status overlay on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Status color
        status_color = (0, 255, 0)  # Green
        if state == PersonState.FALLING:
            status_color = (0, 165, 255)  # Orange
        elif state == PersonState.FALLEN:
            status_color = (0, 0, 255)  # Red
        elif state == PersonState.UNKNOWN:
            status_color = (128, 128, 128)  # Gray
        
        # Draw status text
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
        
        # FPS info
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert overlay for falls
        if state in [PersonState.FALLING, PersonState.FALLEN]:
            alert_overlay = frame.copy()
            cv2.rectangle(alert_overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 255), 3)
            cv2.putText(alert_overlay, "FALL DETECTED!", (w//4 + 20, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(alert_overlay, "Emergency Alert Sent!", (w//4 + 20, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            frame = cv2.addWeighted(alert_overlay, 0.7, frame, 0.3, 0)
        
        return frame
    
    def check_fall_alert(self, state, confidence, timestamp, previous_state):
        """Enhanced fall alert checking with video recording trigger"""
        current_time = time.time()
        
        # Only alert if confidence is above sensitivity threshold
        if confidence < self.sensitivity:
            return
            
        # Check if we should trigger an alert
        if state in [PersonState.FALLING, PersonState.FALLEN]:
            # Check if this is a new fall (transition from non-fall state)
            if previous_state not in [PersonState.FALLING, PersonState.FALLEN]:
                # Avoid spam alerts
                if (current_time - self.last_alert_time) < self.alert_delay:
                    return
                    
                # Create alert
                alert_level = AlertLevel.CRITICAL if state == PersonState.FALLEN else AlertLevel.HIGH
                alert = FallAlert(
                    id=f"alert_{int(timestamp)}",
                    timestamp=datetime.now(),
                    confidence=confidence,
                    state=state,
                    level=alert_level
                )
                
                self.alerts.append(alert)
                self.last_alert_time = current_time
                
                # Trigger callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {str(e)}")
                
                # Log the fall detection event
                logger.info(f"📹 Fall detected! Recording 20s before and 20s after...")
    
    
    def detection_loop(self):
        """Main detection loop"""
        fps_counter = 0
        fps_start_time = time.time()
        self.last_status_emit_time = time.time()
        
        logger.info("Starting fall detection loop")
        
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            try:
                # Process frame
                processed_frame, state, confidence = self.process_frame(frame)
                
                # Update FPS
                fps_counter += 1
                if fps_counter >= 10:
                    elapsed = time.time() - fps_start_time
                    self.fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Store current frame (thread-safe)
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()

                # Emit periodic status updates over WebSocket (once per second)
                now = time.time()
                if now - self.last_status_emit_time >= 1.0:
                    try:
                        status = self.get_status()
                        status_dict = status.__dict__.copy()
                        status_dict['current_state'] = status.current_state.value
                        if 'socketio' in globals() and socketio is not None:
                            socketio.emit('status_update', status_dict)
                    except Exception as emit_error:
                        logger.debug(f"Failed to emit status_update: {emit_error}")
                    finally:
                        self.last_status_emit_time = now
                
            except Exception as e:
                logger.error(f"Error in detection loop: {str(e)}")
                time.sleep(0.1)
                continue
    
    def start_detection(self):
        """Start fall detection"""
        if self.running:
            return True
            
        if not self.initialize_camera():
            return False
            
        self.running = True
        self.start_time = time.time()
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info("Fall detection started")
        return True
    
    def stop_detection(self):
        """Stop fall detection"""
        self.running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
            
        if self.cap:
            self.cap.release()
            
        self.current_frame = None
        logger.info("Fall detection stopped")
    
    def get_current_frame(self):
        """Get current frame (thread-safe)"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        camera_status = "active" if self.running and self.cap is not None else "inactive"
        
        return SystemStatus(
            active=self.running,
            fps=self.fps,
            current_state=self.current_state,
            fall_confidence=self.fall_confidence,
            uptime=uptime,
            total_alerts=len(self.alerts),
            camera_status=camera_status
        )
    
    def add_alert_callback(self, callback):
        """Add callback function for fall alerts"""
        self.alert_callbacks.append(callback)

# Initialize Flask app and extensions
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = Path('fall_recordings')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global detection manager
detection_manager = FallDetectionManager()

# WebSocket event handlers
@socketio.on('connect')
def handle_connect(auth):
    logger.info('Client connected')
    status = detection_manager.get_status()
    status_dict = status.__dict__.copy()
    status_dict['current_state'] = status.current_state.value
    emit('status', status_dict)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

# Alert callback for WebSocket notifications
def alert_callback(alert: FallAlert):
    """Callback function to send alerts via WebSocket"""
    alert_data = {
        'id': alert.id,
        'timestamp': alert.timestamp.isoformat(),
        'confidence': alert.confidence,
        'state': alert.state.value,
        'level': alert.level.value,
        'resolved': alert.resolved,
        'emergency_contacted': alert.emergency_contacted
    }
    socketio.emit('fall_alert', alert_data)
    logger.info(f"Fall alert sent via WebSocket: {alert.id}")

# Add alert callback
detection_manager.add_alert_callback(alert_callback)

# REST API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'fall-detection-api'
    })

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """Start fall detection"""
    try:
        data = request.get_json() or {}
        camera_index = data.get('camera_index', 0)
        
        if detection_manager.running:
            return jsonify({
                'success': False,
                'message': 'Detection is already running'
            }), 400
        
        success = detection_manager.start_detection()
        
        if success:
            # Emit status update via WebSocket
            status = detection_manager.get_status()
            status_dict = status.__dict__.copy()
            status_dict['current_state'] = status.current_state.value
            socketio.emit('status_update', status_dict)
            
            return jsonify({
                'success': True,
                'message': 'Fall detection started successfully',
                'data': status_dict
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start detection. Check camera connection.'
            }), 500
            
    except Exception as e:
        logger.error(f"Error starting detection: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    """Stop fall detection"""
    try:
        detection_manager.stop_detection()
        
        # Emit status update via WebSocket
        status = detection_manager.get_status()
        status_dict = status.__dict__.copy()
        status_dict['current_state'] = status.current_state.value
        socketio.emit('status_update', status_dict)
        
        return jsonify({
            'success': True,
            'message': 'Fall detection stopped successfully',
            'data': status_dict
        })
        
    except Exception as e:
        logger.error(f"Error stopping detection: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/detection/status', methods=['GET'])
def get_detection_status():
    """Get current detection status"""
    try:
        status = detection_manager.get_status()
        status_dict = status.__dict__.copy()
        status_dict['current_state'] = status.current_state.value
        
        return jsonify({
            'success': True,
            'data': status_dict
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/detection/settings', methods=['GET', 'POST'])
def detection_settings():
    """Get or update detection settings"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'data': {
                'sensitivity': detection_manager.sensitivity,
                'alert_delay': detection_manager.alert_delay
            }
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'sensitivity' in data:
                sensitivity = float(data['sensitivity'])
                if 0.1 <= sensitivity <= 1.0:
                    detection_manager.sensitivity = sensitivity
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Sensitivity must be between 0.1 and 1.0'
                    }), 400
            
            if 'alert_delay' in data:
                alert_delay = int(data['alert_delay'])
                if 1 <= alert_delay <= 30:
                    detection_manager.alert_delay = alert_delay
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Alert delay must be between 1 and 30 seconds'
                    }), 400
            
            return jsonify({
                'success': True,
                'message': 'Settings updated successfully',
                'data': {
                    'sensitivity': detection_manager.sensitivity,
                    'alert_delay': detection_manager.alert_delay
                }
            })
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Internal server error: {str(e)}'
            }), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get fall alerts"""
    try:
        # Parse query parameters
        limit = int(request.args.get('limit', 50))
        resolved = request.args.get('resolved')
        level = request.args.get('level')
        
        # Filter alerts
        filtered_alerts = detection_manager.alerts
        
        if resolved is not None:
            resolved_bool = resolved.lower() == 'true'
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved_bool]
        
        if level:
            try:
                level_enum = AlertLevel(level.lower())
                filtered_alerts = [a for a in filtered_alerts if a.level == level_enum]
            except ValueError:
                pass
        
        # Sort by timestamp (newest first) and limit
        filtered_alerts = sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        # Convert to dict
        alerts_data = []
        for alert in filtered_alerts:
            alerts_data.append({
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'confidence': alert.confidence,
                'state': alert.state.value,
                'level': alert.level.value,
                'resolved': alert.resolved,
                'emergency_contacted': alert.emergency_contacted
            })
        
        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts_data,
                'total_count': len(detection_manager.alerts),
                'filtered_count': len(alerts_data)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolve a specific alert"""
    try:
        # Find alert
        alert = next((a for a in detection_manager.alerts if a.id == alert_id), None)
        if not alert:
            return jsonify({
                'success': False,
                'message': 'Alert not found'
            }), 404
        
        # Mark as resolved
        alert.resolved = True
        
        # Emit update via WebSocket
        socketio.emit('alert_resolved', {'alert_id': alert_id})
        
        return jsonify({
            'success': True,
            'message': 'Alert resolved successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resolving alert: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/emergency', methods=['POST'])
def trigger_emergency():
    """Manually trigger emergency alert"""
    try:
        data = request.get_json() or {}
        message = data.get('message', 'Manual emergency alert triggered')
        
        # Create emergency alert
        alert = FallAlert(
            id=f"emergency_{int(time.time())}",
            timestamp=datetime.now(),
            confidence=1.0,
            state=PersonState.FALLEN,
            level=AlertLevel.CRITICAL,
            emergency_contacted=True
        )
        
        detection_manager.alerts.append(alert)
        
        # Trigger alert callback
        alert_callback(alert)
        
        return jsonify({
            'success': True,
            'message': 'Emergency alert triggered successfully',
            'alert_id': alert.id
        })
        
    except Exception as e:
        logger.error(f"Error triggering emergency: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

def generate_video_feed():
    """Generator function for video streaming"""
    while True:
        frame = detection_manager.get_current_frame()
        
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/api/video/stream')
def video_stream():
    """Video streaming endpoint"""
    if not detection_manager.running:
        return jsonify({
            'success': False,
            'message': 'Detection is not active'
        }), 400
    
    return Response(generate_video_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video/frame', methods=['GET'])
def get_current_frame():
    """Get current frame as base64 encoded image"""
    try:
        frame = detection_manager.get_current_frame()
        
        if frame is None:
            return jsonify({
                'success': False,
                'message': 'No frame available. Detection may not be active.'
            }), 404
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return jsonify({
                'success': False,
                'message': 'Failed to encode frame'
            }), 500
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'data': {
                'frame': frame_base64,
                'timestamp': datetime.now().isoformat(),
                'status': detection_manager.current_state.value,
                'confidence': detection_manager.fall_confidence
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting current frame: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    """Get list of recorded fall incidents"""
    try:
        incidents = detection_manager.recorded_incidents
        
        # Sort by timestamp (newest first)
        sorted_incidents = sorted(incidents, 
                                 key=lambda x: x['timestamp'], 
                                 reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'incidents': sorted_incidents,
                'total_count': len(sorted_incidents)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting incidents: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/incidents/<incident_id>/video', methods=['GET'])
def get_incident_video(incident_id):
    """Download a specific incident video"""
    try:
        # Find incident
        incident = next((i for i in detection_manager.recorded_incidents 
                        if i['id'] == incident_id), None)
        if not incident:
            return jsonify({
                'success': False,
                'message': 'Incident not found'
            }), 404
        
        video_path = Path(incident['video_path'])
        if not video_path.exists():
            return jsonify({
                'success': False,
                'message': 'Video file not found'
            }), 404
        
        from flask import send_file
        return send_file(str(video_path), 
                        mimetype='video/mp4',
                        as_attachment=True,
                        download_name=incident['video_filename'])
        
    except Exception as e:
        logger.error(f"Error getting incident video: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500
    
@app.route('/api/test-alert', methods=['POST'])
def test_alert():
    """Trigger a test alert for development purposes"""
    try:
        # Create test alert
        test_alert = FallAlert(
            id=f"test_{int(time.time())}",
            timestamp=datetime.now(),
            confidence=0.85,
            state=PersonState.FALLEN,
            level=AlertLevel.HIGH
        )
        
        detection_manager.alerts.append(test_alert)
        
        # Trigger alert callback
        alert_callback(test_alert)
        
        # Simulate video saved event after a delay
        def delayed_video_event():
            time.sleep(2)
            test_incident = {
                'id': f"test_incident_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'video_filename': 'test_video.mp4',
                'duration': 40,
                'alert_id': test_alert.id
            }
            socketio.emit('video_saved', test_incident)
        
        threading.Thread(target=delayed_video_event, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': 'Test alert triggered',
            'alert_id': test_alert.id
        })
        
    except Exception as e:
        logger.error(f"Error triggering test alert: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500
# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.debug = True
    # Cleanup on exit
    import atexit
    atexit.register(lambda: detection_manager.stop_detection())
    
    # Run the Flask app with SocketIO
    logger.info("Starting Fall Detection Backend Server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)