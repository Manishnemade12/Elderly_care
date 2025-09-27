// Simple test script to verify backend connectivity
const axios = require('axios');

const API_BASE_URL = 'http://localhost:5000';

async function testBackendConnection() {
  console.log('Testing Fall Detection System Integration...\n');
  
  try {
    // Test health endpoint
    console.log('1. Testing health endpoint...');
    const healthResponse = await axios.get(`${API_BASE_URL}/api/health`);
    console.log('✅ Health check passed:', healthResponse.data);
    
    // Test status endpoint
    console.log('\n2. Testing status endpoint...');
    const statusResponse = await axios.get(`${API_BASE_URL}/api/detection/status`);
    console.log('✅ Status check passed:', statusResponse.data);
    
    // Test settings endpoint
    console.log('\n3. Testing settings endpoint...');
    const settingsResponse = await axios.get(`${API_BASE_URL}/api/detection/settings`);
    console.log('✅ Settings check passed:', settingsResponse.data);
    
    console.log('\n🎉 All tests passed! Backend is ready for integration.');
    
  } catch (error) {
    console.error('❌ Backend connection failed:', error.message);
    console.log('\nPlease ensure:');
    console.log('1. Python backend is running: cd udaya && python app.py');
    console.log('2. All Python dependencies are installed: pip install -r requirements.txt');
    console.log('3. Camera is connected and accessible');
  }
}

testBackendConnection();