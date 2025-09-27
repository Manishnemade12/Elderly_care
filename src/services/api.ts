import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface SystemStatus {
  active: boolean;
  fps: number;
  current_state: string;
  fall_confidence: number;
  uptime: number;
  total_alerts: number;
  camera_status: string;
}

export interface FallAlert {
  id: string;
  timestamp: string;
  confidence: number;
  state: string;
  level: string;
  resolved: boolean;
  emergency_contacted: boolean;
}

export interface DetectionSettings {
  sensitivity: number;
  alert_delay: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
}

// API Functions
export const apiService = {
  // Health check
  async healthCheck(): Promise<ApiResponse<any>> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Detection control
  async startDetection(cameraIndex: number = 0): Promise<ApiResponse<SystemStatus>> {
    try {
      const response = await api.post('/detection/start', { camera_index: cameraIndex });
      return response.data;
    } catch (error) {
      console.error('Failed to start detection:', error);
      throw error;
    }
  },

  async stopDetection(): Promise<ApiResponse<SystemStatus>> {
    try {
      const response = await api.post('/detection/stop');
      return response.data;
    } catch (error) {
      console.error('Failed to stop detection:', error);
      throw error;
    }
  },

  // Status
  async getStatus(): Promise<ApiResponse<SystemStatus>> {
    try {
      const response = await api.get('/detection/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get status:', error);
      throw error;
    }
  },

  // Settings
  async getSettings(): Promise<ApiResponse<DetectionSettings>> {
    try {
      const response = await api.get('/detection/settings');
      return response.data;
    } catch (error) {
      console.error('Failed to get settings:', error);
      throw error;
    }
  },

  async updateSettings(settings: Partial<DetectionSettings>): Promise<ApiResponse<DetectionSettings>> {
    try {
      const response = await api.post('/detection/settings', settings);
      return response.data;
    } catch (error) {
      console.error('Failed to update settings:', error);
      throw error;
    }
  },

  // Alerts
  async getAlerts(params?: {
    limit?: number;
    resolved?: boolean;
    level?: string;
  }): Promise<ApiResponse<{ alerts: FallAlert[]; total_count: number; filtered_count: number }>> {
    try {
      const response = await api.get('/alerts', { params });
      return response.data;
    } catch (error) {
      console.error('Failed to get alerts:', error);
      throw error;
    }
  },

  async resolveAlert(alertId: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.post(`/alerts/${alertId}/resolve`);
      return response.data;
    } catch (error) {
      console.error('Failed to resolve alert:', error);
      throw error;
    }
  },

  // Emergency
  async triggerEmergency(message?: string): Promise<ApiResponse<{ alert_id: string }>> {
    try {
      const response = await api.post('/emergency', { message });
      return response.data;
    } catch (error) {
      console.error('Failed to trigger emergency:', error);
      throw error;
    }
  },

  // Video frame
  async getCurrentFrame(): Promise<ApiResponse<{
    frame: string;
    timestamp: string;
    status: string;
    confidence: number;
  }>> {
    try {
      const response = await api.get('/video/frame');
      return response.data;
    } catch (error) {
      console.error('Failed to get current frame:', error);
      throw error;
    }
  },

  // Video stream URL
  getVideoStreamUrl(): string {
    return `${API_BASE_URL}/api/video/stream`;
  }
};