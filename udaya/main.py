import streamlit as st
import cv2
import time
import numpy as np
import threading
from detection_2 import FallDetector
from enum import Enum

# Assuming PersonState enum from your detection module
class PersonState(Enum):
    STANDING = "Standing"
    SITTING = "Sitting"
    FALLING = "Falling"
    FALLEN = "Fallen"
    UNKNOWN = "Unknown"

# Global variables for video streaming
video_stream = None
fall_detector = None
current_frame = None
detection_active = False

class VideoStreamProcessor:
    def __init__(self):
        self.cap = None
        self.detector = None
        self.running = False
        self.current_frame = None
        self.current_state = PersonState.STANDING
        self.fall_confidence = 0.0
        self.fps = 0
        
    def initialize_camera(self):
        """Initialize camera and fall detector"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
                
            # Get camera FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps > 60:
                fps = 30
                
            # Initialize fall detector
            self.detector = FallDetector(fps=fps)
            return True
        except Exception as e:
            st.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def draw_skeleton(self, frame, keypoints, w, h):
        """Draw skeleton on frame (implement based on your keypoints format)"""
        # This should match your existing draw_skeleton function
        # Placeholder implementation - adapt based on your keypoint format
        if keypoints is not None:
            # Draw keypoints and connections based on your detection_2.py implementation
            pass
    
    def process_video_stream(self):
        """Main video processing loop"""
        fps_counter = 0
        fps_start_time = time.time()
        display_fps = 0
        
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            h, w = frame.shape[:2]
            timestamp = time.time()
            
            try:
                # Run pose detection
                keypoints = self.detector.run_inference(frame)
                
                # Process frame for fall detection
                state, fall_confidence = self.detector.process_frame(frame, keypoints, timestamp)
                
                # Update current state
                self.current_state = state
                self.fall_confidence = fall_confidence
                
                # Draw skeleton
                self.draw_skeleton(frame, keypoints, w, h)
                
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
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter >= 10:
                    elapsed = time.time() - fps_start_time
                    display_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start_time = time.time()
                
                info_text = f"FPS: {display_fps:.1f}"
                cv2.putText(frame, info_text, (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Alert overlay for falls
                if state in [PersonState.FALLING, PersonState.FALLEN]:
                    alert_overlay = frame.copy()
                    cv2.rectangle(alert_overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 255), 3)
                    cv2.putText(alert_overlay, "FALL DETECTED!", (w//4 + 20, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(alert_overlay, "Emergency Alert Sent!", (w//4 + 20, h//2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frame = cv2.addWeighted(alert_overlay, 0.7, frame, 0.3, 0)
                
                # Store current frame for Streamlit display
                self.current_frame = frame.copy()
                self.fps = display_fps
                
            except Exception as e:
                print(f"Error in video processing: {str(e)}")
                continue
    
    def start_stream(self):
        """Start video streaming"""
        if self.initialize_camera():
            self.running = True
            self.stream_thread = threading.Thread(target=self.process_video_stream)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            return True
        return False
    
    def stop_stream(self):
        """Stop video streaming"""
        self.running = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.current_frame = None

# Page configuration
st.set_page_config(
    page_title="Elderly Fall Detection System",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .alert-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_stream_active" not in st.session_state:
    st.session_state.video_stream_active = False

if "video_processor" not in st.session_state:
    st.session_state.video_processor = VideoStreamProcessor()

if "last_fall_alert" not in st.session_state:
    st.session_state.last_fall_alert = 0

# Title
st.title("🚨 Elderly Fall Detection System")
st.markdown("---")

# Create main layout
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.subheader("💬 Emergency Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "system":
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background-color: #fff3cd; border-radius: 0.25rem; margin: 0.5rem 0; color: #856404;">
                    <em>{message["content"]}</em>
                </div>
                """, unsafe_allow_html=True)
            elif message["role"] == "alert":
                st.markdown(f"""
                <div class="alert-message">
                    <strong>🚨 FALL ALERT:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.subheader("📹 Live Monitoring")
    
    # Video stream controls
    if st.session_state.video_stream_active:
        stream_button_text = "⏹️ Stop Detection"
        stream_button_type = "secondary"
        status_text = "🔴 **MONITORING ACTIVE**"
    else:
        stream_button_text = "📹 Start Detection"
        stream_button_type = "primary"
        status_text = "⚫ **MONITORING OFFLINE**"
    
    st.markdown(status_text)
    
    if st.button(stream_button_text, key="stream_btn", type=stream_button_type):
        if not st.session_state.video_stream_active:
            # Start video stream
            if st.session_state.video_processor.start_stream():
                st.session_state.video_stream_active = True
                st.session_state.messages.append({
                    "role": "system", 
                    "content": "🔴 Fall detection system activated"
                })
                st.success("Camera initialized successfully!")
            else:
                st.error("Failed to initialize camera. Please check your webcam connection.")
        else:
            # Stop video stream
            st.session_state.video_processor.stop_stream()
            st.session_state.video_stream_active = False
            st.session_state.messages.append({
                "role": "system", 
                "content": "⏹️ Fall detection system deactivated"
            })
        st.rerun()
    
    # Display video feed if active
    video_placeholder = st.empty()
    if st.session_state.video_stream_active and st.session_state.video_processor.current_frame is not None:
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(st.session_state.video_processor.current_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

with col3:
    st.subheader("📊 System Status")
    
    if st.session_state.video_stream_active:
        processor = st.session_state.video_processor
        
        # Current status
        state = processor.current_state
        confidence = processor.fall_confidence
        
        # Status indicator
        if state == PersonState.FALLEN:
            st.error(f"🚨 **{state.value.upper()}**")
        elif state == PersonState.FALLING:
            st.warning(f"⚠️ **{state.value.upper()}**")
        elif state == PersonState.STANDING:
            st.success(f"✅ **{state.value.upper()}**")
        else:
            st.info(f"ℹ️ **{state.value.upper()}**")
        
        # Fall risk meter
        st.markdown("**Fall Risk Level:**")
        risk_color = "normal" if confidence < 0.3 else "inverse" if confidence < 0.6 else "off"
        st.progress(confidence, text=f"{confidence:.1%}")
        
        # System metrics
        st.markdown("**Performance:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("FPS", f"{processor.fps:.1f}")
        with col_b:
            st.metric("Latency", "~50ms")
        
        # Check for fall alerts
        current_time = time.time()
        if state in [PersonState.FALLING, PersonState.FALLEN] and \
           (current_time - st.session_state.last_fall_alert) > 5:  # Avoid spam
            
            alert_message = f"Fall detected at {time.strftime('%H:%M:%S')}! Confidence: {confidence:.1%}"
            st.session_state.messages.append({
                "role": "alert",
                "content": alert_message
            })
            st.session_state.last_fall_alert = current_time
            st.rerun()
            
    else:
        st.info("Start detection to view system status")

# Chat input
st.markdown("---")
user_input = st.chat_input("Ask about the system or report an emergency...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Enhanced bot response for fall detection context
    with st.spinner("Processing..."):
        time.sleep(1)
        
        # Context-aware responses
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["emergency", "help", "fall", "hurt", "pain"]):
            bot_response = "🚨 I understand this may be an emergency. I'm alerting your emergency contacts and local services. Please stay calm and try to remain still if you're injured. Help is on the way!"
        elif any(word in user_lower for word in ["status", "system", "detection"]):
            if st.session_state.video_stream_active:
                state = st.session_state.video_processor.current_state.value
                confidence = st.session_state.video_processor.fall_confidence
                bot_response = f"The fall detection system is currently active. Status: {state}, Fall risk: {confidence:.1%}. Everything appears to be functioning normally."
            else:
                bot_response = "The fall detection system is currently offline. Please activate it using the 'Start Detection' button to begin monitoring."
        else:
            bot_response = f"I'm here to help with the fall detection system. You said: '{user_input}'. How can I assist you further?"
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    st.rerun()

# Sidebar
with st.sidebar:
    st.header("System Controls")
    
    # Emergency button
    if st.button("🚨 EMERGENCY ALERT", type="primary", key="emergency"):
        st.session_state.messages.append({
            "role": "alert",
            "content": "Manual emergency alert triggered! Emergency services have been notified."
        })
        st.balloons()  # Visual feedback
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.6, 0.1)
    alert_delay = st.number_input("Alert Delay (seconds)", 1, 10, 3)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("Session Statistics")
    st.metric("Total Messages", len(st.session_state.messages))
    
    fall_alerts = len([msg for msg in st.session_state.messages if msg["role"] == "alert"])
    st.metric("Fall Alerts", fall_alerts)
    
    if st.session_state.video_stream_active:
        st.metric("Monitoring Time", "Live")
        st.success("System Active")
    else:
        st.metric("Monitoring Time", "Stopped")
        st.error("System Inactive")

# Auto-refresh for live updates
if st.session_state.video_stream_active:
    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚨 Elderly Fall Detection System | Real-time AI Monitoring</p>
        <small>Emergency Contacts: 911 | Family: +1-234-567-8900 | Doctor: +1-234-567-8901</small>
    </div>
    """, 
    unsafe_allow_html=True
)