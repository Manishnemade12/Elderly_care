# AI Fall Detection System

A comprehensive fall detection system that combines computer vision AI with a modern web interface for real-time monitoring and emergency response.

## Features

- **Real-time Fall Detection**: Advanced AI model using pose estimation to detect falls
- **Live Video Streaming**: Real-time camera feed with overlay information
- **WebSocket Integration**: Instant alerts and status updates
- **Emergency Response**: Automated alert system with emergency protocols
- **Interactive Chat Assistant**: AI-powered chatbot for system interaction
- **Event Logging**: Comprehensive logging of all system activities
- **Responsive UI**: Modern React interface with shadcn/ui components

## System Architecture

### Backend (Python)
- **Flask API**: RESTful API for system control
- **WebSocket Support**: Real-time communication using Socket.IO
- **Computer Vision**: TensorFlow/OpenCV for fall detection
- **Pose Estimation**: Advanced ML model for human pose analysis

### Frontend (React)
- **Modern UI**: Built with React, TypeScript, and Tailwind CSS
- **Real-time Updates**: WebSocket integration for live data
- **Responsive Design**: Works on desktop and mobile devices
- **Component Library**: shadcn/ui for consistent design

## Prerequisites

### Backend Requirements
- Python 3.8+
- OpenCV
- TensorFlow
- Flask
- Socket.IO

### Frontend Requirements
- Node.js 16+
- npm or yarn

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd fall-detection-system
```

### 2. Backend Setup
```bash
cd udaya
pip install -r requirements.txt
```

Create a `requirements.txt` file in the `udaya` directory:
```
flask==2.3.3
flask-cors==4.0.0
flask-socketio==5.3.6
opencv-python==4.8.1.78
tensorflow==2.13.0
tensorflow-hub==0.14.0
numpy==1.24.3
python-socketio==5.8.0
```

### 3. Frontend Setup
```bash
npm install
```

## Running the System

### Option 1: Automatic Startup (Windows)
```bash
start-system.bat
```

### Option 2: Manual Startup

#### Start Backend
```bash
cd udaya
python app.py
```

#### Start Frontend
```bash
npm run dev
```

## Usage

1. **Access the Interface**: Open http://localhost:5173 in your browser
2. **Start Detection**: Click "Start Detection" to begin monitoring
3. **Monitor Status**: View real-time status in the status panel
4. **Check Events**: Review system events in the event log
5. **Chat Assistant**: Use the chat bot for help and emergency assistance

## API Endpoints

### Detection Control
- `POST /api/detection/start` - Start fall detection
- `POST /api/detection/stop` - Stop fall detection
- `GET /api/detection/status` - Get current status

### Alerts
- `GET /api/alerts` - Get fall alerts
- `POST /api/alerts/{id}/resolve` - Resolve specific alert
- `POST /api/emergency` - Trigger emergency alert

### Video
- `GET /api/video/stream` - Live video stream
- `GET /api/video/frame` - Get current frame

### Settings
- `GET /api/detection/settings` - Get detection settings
- `POST /api/detection/settings` - Update settings

## WebSocket Events

### Client → Server
- `connect` - Establish connection

### Server → Client
- `status` - System status updates
- `fall_alert` - Fall detection alerts
- `status_update` - Real-time status changes
- `alert_resolved` - Alert resolution notifications

## Configuration

### Environment Variables
Create a `.env` file in the root directory:
```
VITE_API_URL=http://localhost:5000
VITE_DEV_MODE=true
```

### Detection Settings
- **Sensitivity**: Adjust fall detection sensitivity (0.1 - 1.0)
- **Alert Delay**: Time delay between alerts (1-30 seconds)

## Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Ensure camera is connected and not used by other applications
   - Check camera permissions
   - Try different camera index (0, 1, 2...)

2. **Backend Connection Failed**
   - Verify Python backend is running on port 5000
   - Check firewall settings
   - Ensure all Python dependencies are installed

3. **WebSocket Connection Issues**
   - Check if port 5000 is available
   - Verify CORS settings
   - Try refreshing the browser

4. **Video Stream Not Loading**
   - Check camera permissions
   - Verify backend is running
   - Try the frame-by-frame fallback mode

## Development

### Project Structure
```
├── src/
│   ├── components/     # React components
│   ├── hooks/         # Custom React hooks
│   ├── services/      # API and WebSocket services
│   └── pages/         # Page components
├── udaya/             # Python backend
│   ├── app.py         # Main Flask application
│   ├── detection_2.py # Fall detection logic
│   └── ...
└── public/            # Static assets
```

### Adding New Features

1. **Backend**: Add new endpoints in `udaya/app.py`
2. **Frontend**: Create components in `src/components/`
3. **API Integration**: Update `src/services/api.ts`
4. **WebSocket Events**: Modify `src/services/websocket.ts`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Check the troubleshooting section
- Use the in-app chat assistant
- Review system logs in the event panel
- 



The problem ELDER CARE solves

The primary problem is the high risk of injury and delayed assistance following falls among the elderly, which can lead to serious health complications, prolonged recovery, or even fatal outcomes. Many older adults live alone or experience limited supervision, making timely detection and intervention critical. Traditional monitoring methods rely heavily on manual checks or wearable devices, which may be unreliable or go unnoticed during an emergency. This creates an urgent need for an automated, real-time fall detection system that ensures immediate alerts and rapid response to safeguard elderly individuals’ health and independence.
