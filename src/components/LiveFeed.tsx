import { useState, useEffect } from "react";
import { apiService, SystemStatus } from "@/services/api";

interface LiveFeedProps {
  isDetectionActive: boolean;
  systemStatus: SystemStatus | null;
}

const LiveFeed = ({ isDetectionActive, systemStatus }: LiveFeedProps) => {
  const [imageError, setImageError] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);

  // Get video stream URL
  const streamUrl = apiService.getVideoStreamUrl();

  // Fallback to frame-by-frame updates if stream fails
  useEffect(() => {
    if (!isDetectionActive || !imageError) return;

    const interval = setInterval(async () => {
      try {
        const response = await apiService.getCurrentFrame();
        if (response.success && response.data) {
          setCurrentFrame(`data:image/jpeg;base64,${response.data.frame}`);
        }
      } catch (error) {
        console.error('Failed to get current frame:', error);
      }
    }, 100); // 10 FPS fallback

    return () => clearInterval(interval);
  }, [isDetectionActive, imageError]);

  const getStatusColor = () => {
    if (!systemStatus) return "bg-gray-500";
    
    switch (systemStatus.current_state) {
      case "Fallen":
        return "bg-red-600";
      case "Falling":
        return "bg-orange-500";
      case "Standing":
      case "Sitting":
        return "bg-green-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusText = () => {
    if (!isDetectionActive) return "OFFLINE";
    if (!systemStatus) return "CONNECTING";
    return "LIVE";
  };

  return (
    <div className="h-full">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-foreground flex items-center space-x-2">
          <div className={`w-3 h-3 ${getStatusColor()} rounded-full ${isDetectionActive ? 'animate-pulse' : ''}`}></div>
          <span>Live Camera Feed</span>
          {systemStatus && (
            <span className="text-sm text-muted-foreground">
              ({systemStatus.fps.toFixed(1)} FPS)
            </span>
          )}
        </h2>
      </div>
      
      <div className="relative bg-video-bg rounded-lg shadow-lg border border-border h-[calc(100%-4rem)] min-h-96 overflow-hidden">
        {isDetectionActive ? (
          <>
            {/* Primary video stream */}
            <img
              src={streamUrl}
              alt="Live video feed from AI Fall Detection camera"
              className={`w-full h-full object-cover rounded-lg ${imageError ? 'hidden' : ''}`}
              onError={() => setImageError(true)}
              onLoad={() => setImageError(false)}
            />
            
            {/* Fallback frame display */}
            {imageError && currentFrame && (
              <img
                src={currentFrame}
                alt="Current frame from fall detection"
                className="w-full h-full object-cover rounded-lg"
              />
            )}
          </>
        ) : null}
        
        {/* Placeholder content when detection is not active or stream unavailable */}
        {(!isDetectionActive || (imageError && !currentFrame)) && (
          <div className="absolute inset-0 flex items-center justify-center text-white/70 bg-gray-900/50">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 mx-auto border-2 border-white/30 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 002 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <p className="text-xl font-medium">
                  {!isDetectionActive ? "Detection Offline" : "Connecting to Camera Feed..."}
                </p>
                <p className="text-sm mt-2">
                  {!isDetectionActive 
                    ? "Start detection to begin monitoring" 
                    : "Please ensure the backend is running"
                  }
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* Status overlays */}
        {isDetectionActive && (
          <>
            {/* Live indicator */}
            <div className={`absolute top-4 left-4 ${getStatusColor()} text-white px-3 py-1 rounded-full text-sm font-medium flex items-center space-x-2`}>
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span>{getStatusText()}</span>
            </div>
            
            {/* Person status */}
            {systemStatus && (
              <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded text-sm">
                Status: {systemStatus.current_state}
              </div>
            )}
            
            {/* Fall confidence indicator */}
            {systemStatus && systemStatus.fall_confidence > 0.3 && (
              <div className="absolute top-16 right-4 bg-red-600/90 text-white px-3 py-1 rounded text-sm font-bold">
                Fall Risk: {(systemStatus.fall_confidence * 100).toFixed(0)}%
              </div>
            )}
            
            {/* Timestamp overlay */}
            <div className="absolute bottom-4 right-4 bg-black/50 text-white px-3 py-1 rounded text-sm font-mono">
              {new Date().toLocaleTimeString()}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LiveFeed;