import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, AlertTriangle, Info, CheckCircle, AlertCircle, Video, Download } from 'lucide-react';

const EventLog = ({ apiUrl = 'http://localhost:5000' }) => {
  const [events, setEvents] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [expandedEvents, setExpandedEvents] = useState(new Set());

  // Initialize WebSocket connection
  useEffect(() => {
    let ws = null;
    
    const connectWebSocket = () => {
      // Using native WebSocket for simplicity (you can use socket.io-client if preferred)
      ws = new WebSocket(`ws://localhost:5000/socket.io/?EIO=4&transport=websocket`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        addEvent('info', 'Connected to fall detection system');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        addEvent('warning', 'Disconnected from fall detection system');
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
    };
    
    // For demo purposes, use polling instead if WebSocket is complex
    // This is a simpler approach that polls the API
    const pollAlerts = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/alerts?limit=10`);
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data.alerts) {
            processAlerts(data.data.alerts);
          }
        }
      } catch (err) {
        console.error('Error polling alerts:', err);
      }
    };
    
    const pollIncidents = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/incidents`);
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data.incidents) {
            setIncidents(data.data.incidents);
          }
        }
      } catch (err) {
        console.error('Error polling incidents:', err);
      }
    };
    
    // Start polling (simpler than WebSocket for demo)
    const pollInterval = setInterval(() => {
      pollAlerts();
      pollIncidents();
    }, 2000);
    
    // Initial fetch
    pollAlerts();
    pollIncidents();
    
    return () => {
      if (ws) ws.close();
      clearInterval(pollInterval);
    };
  }, [apiUrl]);
  
  const handleWebSocketMessage = (data) => {
    if (data.type === 'fall_alert') {
      handleFallAlert(data.data);
    } else if (data.type === 'video_saved') {
      handleVideoSaved(data.data);
    } else if (data.type === 'status_update') {
      handleStatusUpdate(data.data);
    }
  };
  
  const processAlerts = (alerts) => {
    alerts.forEach(alert => {
      const eventId = `alert_${alert.id}`;
      const existingEvent = events.find(e => e.id === eventId);
      if (!existingEvent) {
        handleFallAlert(alert);
      }
    });
  };
  
  const handleFallAlert = (alertData) => {
    const message = `Fall detected with ${(alertData.confidence * 100).toFixed(0)}% confidence. Person is ${alertData.state}. Recording video...`;
    addEvent('alert', message, {
      alertId: alertData.id,
      confidence: alertData.confidence,
      state: alertData.state,
      level: alertData.level
    });
  };
  
  const handleVideoSaved = (videoData) => {
    const message = `Video recording complete for incident. Duration: ${videoData.duration}s (20s before + 20s after fall)`;
    addEvent('info', message, {
      incidentId: videoData.id,
      videoPath: videoData.video_filename,
      alertId: videoData.alert_id
    });
  };
  
  const handleStatusUpdate = (status) => {
    // You can use this to update UI based on system status
    console.log('Status update:', status);
  };
  
  const addEvent = (type, message, metadata = {}) => {
    const newEvent = {
      id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      }),
      message,
      type,
      metadata
    };
    
    setEvents(prev => [newEvent, ...prev].slice(0, 100)); // Keep last 100 events
  };
  
  const getEventIcon = (type) => {
    switch (type) {
      case 'alert':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'normal':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'info':
      default:
        return <Info className="w-5 h-5 text-blue-500" />;
    }
  };
  
  const getEventColor = (type) => {
    switch (type) {
      case 'alert':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'normal':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'info':
      default:
        return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };
  
  const toggleEventExpansion = (eventId) => {
    setExpandedEvents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(eventId)) {
        newSet.delete(eventId);
      } else {
        newSet.add(eventId);
      }
      return newSet;
    });
  };
  
  const downloadVideo = async (incidentId) => {
    try {
      window.open(`${apiUrl}/api/incidents/${incidentId}/video`, '_blank');
    } catch (err) {
      console.error('Error downloading video:', err);
    }
  };
  
  const triggerTestAlert = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/test-alert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        addEvent('info', 'Test alert triggered successfully');
      }
    } catch (err) {
      console.error('Error triggering test alert:', err);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200">
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
            <span>Event Log</span>
            <div className={`ml-2 w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={triggerTestAlert}
              className="text-xs bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 transition-colors"
            >
              Test Alert
            </button>
            <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {events.length} events
            </div>
          </div>
        </div>
      </div>
      
      <div className="h-96 overflow-y-auto">
        <div className="p-4 space-y-2">
          {events.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Info className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No events recorded yet</p>
              <p className="text-xs mt-1">Events will appear here when fall detection is active</p>
            </div>
          ) : (
            events.map((event) => (
              <div
                key={event.id}
                className={`border rounded-lg transition-all ${getEventColor(event.type)} ${
                  expandedEvents.has(event.id) ? 'shadow-md' : ''
                }`}
              >
                <div
                  className="flex items-start space-x-3 p-3 cursor-pointer"
                  onClick={() => toggleEventExpansion(event.id)}
                >
                  <span className="flex-shrink-0 mt-0.5">
                    {getEventIcon(event.type)}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-xs font-mono bg-white px-2 py-0.5 rounded">
                        {event.timestamp}
                      </span>
                      {event.type === 'alert' && (
                        <span className="text-xs bg-red-500 text-white px-2 py-0.5 rounded animate-pulse">
                          FALL DETECTED
                        </span>
                      )}
                    </div>
                    <p className="text-sm font-medium">
                      {event.message}
                    </p>
                  </div>
                  {event.metadata && Object.keys(event.metadata).length > 0 && (
                    <ChevronDown 
                      className={`w-4 h-4 transition-transform ${
                        expandedEvents.has(event.id) ? 'rotate-180' : ''
                      }`}
                    />
                  )}
                </div>
                
                {expandedEvents.has(event.id) && event.metadata && (
                  <div className="px-3 pb-3 border-t">
                    <div className="mt-2 text-xs space-y-1">
                      {event.metadata.confidence && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Confidence:</span>
                          <span className="font-mono">{(event.metadata.confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {event.metadata.state && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">State:</span>
                          <span className="font-mono">{event.metadata.state}</span>
                        </div>
                      )}
                      {event.metadata.videoPath && (
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-gray-600 flex items-center">
                            <Video className="w-3 h-3 mr-1" />
                            Video saved
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              downloadVideo(event.metadata.incidentId);
                            }}
                            className="flex items-center space-x-1 bg-blue-500 text-white px-2 py-1 rounded text-xs hover:bg-blue-600"
                          >
                            <Download className="w-3 h-3" />
                            <span>Download</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
      
      {incidents.length > 0 && (
        <div className="border-t bg-gray-50 p-3">
          <div className="text-xs text-gray-600">
            <strong>Recent Incidents:</strong> {incidents.length} recorded
            {incidents[0] && (
              <span className="ml-2">
                Last: {new Date(incidents[0].timestamp).toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EventLog;