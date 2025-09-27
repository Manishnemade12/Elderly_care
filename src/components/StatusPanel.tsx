import { Button } from "@/components/ui/button";
import { SystemStatus } from "@/services/api";
import { AlertTriangle, Play, Square, Zap } from "lucide-react";

interface StatusPanelProps {
  systemStatus: SystemStatus | null;
  isDetectionActive: boolean;
  loading: boolean;
  error: string | null;
  onStartDetection: () => void;
  onStopDetection: () => void;
  onTriggerEmergency: () => void;
}

const StatusPanel = ({ 
  systemStatus, 
  isDetectionActive, 
  loading, 
  error,
  onStartDetection, 
  onStopDetection, 
  onTriggerEmergency 
}: StatusPanelProps) => {
  const getStatusDisplay = () => {
    if (!isDetectionActive) {
      return {
        text: "SYSTEM OFFLINE",
        color: "bg-gray-500 text-white border-gray-500",
        icon: "⏸️"
      };
    }
    
    if (!systemStatus) {
      return {
        text: "CONNECTING...",
        color: "bg-yellow-500 text-white border-yellow-500",
        icon: "🔄"
      };
    }

    switch (systemStatus.current_state) {
      case "Fallen":
        return {
          text: "⚠️ FALL DETECTED!",
          color: "bg-red-600 text-white border-red-600 animate-pulse",
          icon: "🚨"
        };
      case "Falling":
        return {
          text: "⚠️ FALL IN PROGRESS",
          color: "bg-orange-500 text-white border-orange-500 animate-pulse",
          icon: "⚠️"
        };
      case "Standing":
        return {
          text: "✓ STANDING - NORMAL",
          color: "bg-green-600 text-white border-green-600",
          icon: "✅"
        };
      case "Sitting":
        return {
          text: "✓ SITTING - NORMAL",
          color: "bg-blue-500 text-white border-blue-500",
          icon: "💺"
        };
      default:
        return {
          text: "STATUS: MONITORING",
          color: "bg-blue-600 text-white border-blue-600",
          icon: "👁️"
        };
    }
  };

  const status = getStatusDisplay();
  const isAlert = systemStatus?.current_state === "Fallen" || systemStatus?.current_state === "Falling";

  return (
    <div className="space-y-6">
      {/* Status Indicator */}
      <div className="bg-card rounded-lg border border-border p-6 shadow-lg">
        <h3 className="text-lg font-semibold text-foreground mb-4">System Status</h3>
        
        <div className={`rounded-lg p-4 text-center font-bold text-lg border-2 transition-all duration-300 ${status.color}`}>
          {status.text}
        </div>
        
        {/* Fall confidence meter */}
        {systemStatus && isDetectionActive && (
          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Fall Risk Level:</span>
              <span className="font-medium">{(systemStatus.fall_confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${
                  systemStatus.fall_confidence > 0.7 ? 'bg-red-500' :
                  systemStatus.fall_confidence > 0.4 ? 'bg-orange-500' : 'bg-green-500'
                }`}
                style={{ width: `${systemStatus.fall_confidence * 100}%` }}
              ></div>
            </div>
          </div>
        )}
        
        {isAlert && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800 font-medium flex items-center">
              <AlertTriangle className="w-4 h-4 mr-2" />
              Emergency protocols activated. Check on individual immediately.
            </p>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}
        
        {/* Control Buttons */}
        <div className="mt-6 pt-4 border-t border-border space-y-3">
          <div className="flex space-x-2">
            {!isDetectionActive ? (
              <Button 
                onClick={() => onStartDetection(0)}
                disabled={loading}
                className="flex-1"
                size="sm"
              >
                <Play className="w-4 h-4 mr-2" />
                {loading ? "Starting..." : "Start Detection"}
              </Button>
            ) : (
              <Button 
                onClick={onStopDetection}
                disabled={loading}
                variant="outline"
                className="flex-1"
                size="sm"
              >
                <Square className="w-4 h-4 mr-2" />
                {loading ? "Stopping..." : "Stop Detection"}
              </Button>
            )}
          </div>
          
          <Button 
            onClick={() => onTriggerEmergency()}
            variant="destructive"
            size="sm"
            className="w-full"
          >
            <Zap className="w-4 h-4 mr-2" />
            Emergency Alert
          </Button>
        </div>
      </div>
      
      {/* System Information */}
      <div className="bg-card rounded-lg border border-border p-4 shadow-lg">
        <h4 className="font-semibold text-foreground mb-3">System Info</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Camera Status:</span>
            <span className={`font-medium ${
              systemStatus?.camera_status === 'active' ? 'text-green-600' : 'text-red-600'
            }`}>
              {systemStatus?.camera_status || 'Unknown'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">AI Model:</span>
            <span className={`font-medium ${isDetectionActive ? 'text-green-600' : 'text-gray-500'}`}>
              {isDetectionActive ? 'Online' : 'Offline'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">FPS:</span>
            <span className="text-foreground font-mono">
              {systemStatus?.fps?.toFixed(1) || '0.0'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Uptime:</span>
            <span className="text-foreground font-mono">
              {systemStatus ? `${Math.floor(systemStatus.uptime / 60)}m ${Math.floor(systemStatus.uptime % 60)}s` : '0s'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Total Alerts:</span>
            <span className="text-foreground font-medium">
              {systemStatus?.total_alerts || 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatusPanel;