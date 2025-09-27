import { useState, useEffect, useCallback } from 'react';
import { apiService, SystemStatus, FallAlert, DetectionSettings } from '@/services/api';
import { websocketService } from '@/services/websocket';

export interface LogEvent {
  id: string;
  timestamp: string;
  message: string;
  type: 'normal' | 'alert' | 'warning' | 'info';
}

export const useFallDetection = () => {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [alerts, setAlerts] = useState<FallAlert[]>([]);
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [settings, setSettings] = useState<DetectionSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add event to log
  const addEvent = useCallback((message: string, type: LogEvent['type'] = 'info') => {
    const event: LogEvent = {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleTimeString(),
      message,
      type,
    };
    setEvents(prev => [event, ...prev.slice(0, 49)]); // Keep last 50 events
  }, []);

  // Initialize WebSocket connection
  const connectWebSocket = useCallback(async () => {
    try {
      await websocketService.connect();
      setIsConnected(true);
      addEvent('WebSocket connected', 'info');

      // Set up event listeners
      websocketService.on('status', (status) => {
        setSystemStatus(status);
      });

      websocketService.on('status_update', (status) => {
        setSystemStatus(status);
        setIsDetectionActive(status.active);
        addEvent(`System status updated: ${status.active ? 'Active' : 'Inactive'}`, 'info');
      });

      websocketService.on('fall_alert', (alert) => {
        setAlerts(prev => [alert, ...prev]);
        addEvent(`Fall detected! Confidence: ${(alert.confidence * 100).toFixed(1)}%`, 'alert');
      });

      websocketService.on('alert_resolved', (data) => {
        setAlerts(prev => 
          prev.map(alert => 
            alert.id === data.alert_id 
              ? { ...alert, resolved: true }
              : alert
          )
        );
        addEvent(`Alert ${data.alert_id} resolved`, 'normal');
      });

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setIsConnected(false);
      addEvent('WebSocket connection failed', 'warning');
    }
  }, [addEvent]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    websocketService.disconnect();
    setIsConnected(false);
    addEvent('WebSocket disconnected', 'info');
  }, [addEvent]);

  // Start detection
  const startDetection = useCallback(async (cameraIndex: number = 0) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.startDetection(cameraIndex);
      if (response.success && response.data) {
        setSystemStatus(response.data);
        setIsDetectionActive(true);
        addEvent('Fall detection started', 'normal');
      } else {
        throw new Error(response.message || 'Failed to start detection');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      addEvent(`Failed to start detection: ${errorMessage}`, 'warning');
    } finally {
      setLoading(false);
    }
  }, [addEvent]);

  // Stop detection
  const stopDetection = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.stopDetection();
      if (response.success && response.data) {
        setSystemStatus(response.data);
        setIsDetectionActive(false);
        addEvent('Fall detection stopped', 'normal');
      } else {
        throw new Error(response.message || 'Failed to stop detection');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      addEvent(`Failed to stop detection: ${errorMessage}`, 'warning');
    } finally {
      setLoading(false);
    }
  }, [addEvent]);

  // Get current status
  const refreshStatus = useCallback(async () => {
    try {
      const response = await apiService.getStatus();
      if (response.success && response.data) {
        setSystemStatus(response.data);
        setIsDetectionActive(response.data.active);
      }
    } catch (error) {
      console.error('Failed to refresh status:', error);
    }
  }, []);

  // Load alerts
  const loadAlerts = useCallback(async () => {
    try {
      const response = await apiService.getAlerts({ limit: 50 });
      if (response.success && response.data) {
        setAlerts(response.data.alerts);
      }
    } catch (error) {
      console.error('Failed to load alerts:', error);
    }
  }, []);

  // Resolve alert
  const resolveAlert = useCallback(async (alertId: string) => {
    try {
      const response = await apiService.resolveAlert(alertId);
      if (response.success) {
        setAlerts(prev => 
          prev.map(alert => 
            alert.id === alertId 
              ? { ...alert, resolved: true }
              : alert
          )
        );
        addEvent(`Alert ${alertId} resolved`, 'normal');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addEvent(`Failed to resolve alert: ${errorMessage}`, 'warning');
    }
  }, [addEvent]);

  // Trigger emergency
  const triggerEmergency = useCallback(async (message?: string) => {
    try {
      const response = await apiService.triggerEmergency(message);
      if (response.success && response.data) {
        addEvent(`Emergency alert triggered: ${response.data.alert_id}`, 'alert');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addEvent(`Failed to trigger emergency: ${errorMessage}`, 'warning');
    }
  }, [addEvent]);

  // Load settings
  const loadSettings = useCallback(async () => {
    try {
      const response = await apiService.getSettings();
      if (response.success && response.data) {
        setSettings(response.data);
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }, []);

  // Update settings
  const updateSettings = useCallback(async (newSettings: Partial<DetectionSettings>) => {
    try {
      const response = await apiService.updateSettings(newSettings);
      if (response.success && response.data) {
        setSettings(response.data);
        addEvent('Settings updated', 'info');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addEvent(`Failed to update settings: ${errorMessage}`, 'warning');
    }
  }, [addEvent]);

  // Initialize on mount
  useEffect(() => {
    const initialize = async () => {
      // Check backend health
      try {
        await apiService.healthCheck();
        addEvent('Backend connection established', 'normal');
        
        // Load initial data
        await Promise.all([
          refreshStatus(),
          loadAlerts(),
          loadSettings(),
        ]);
        
        // Connect WebSocket
        await connectWebSocket();
      } catch (error) {
        addEvent('Failed to connect to backend', 'warning');
        setError('Backend connection failed. Please ensure the Python backend is running.');
      }
    };

    initialize();

    // Cleanup on unmount
    return () => {
      disconnectWebSocket();
    };
  }, [connectWebSocket, disconnectWebSocket, refreshStatus, loadAlerts, loadSettings, addEvent]);

  // Periodic status refresh when not connected via WebSocket
  useEffect(() => {
    if (!isConnected && isDetectionActive) {
      const interval = setInterval(refreshStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [isConnected, isDetectionActive, refreshStatus]);

  return {
    // State
    isConnected,
    isDetectionActive,
    systemStatus,
    alerts,
    events,
    settings,
    loading,
    error,
    
    // Actions
    startDetection,
    stopDetection,
    refreshStatus,
    resolveAlert,
    triggerEmergency,
    updateSettings,
    connectWebSocket,
    disconnectWebSocket,
  };
};