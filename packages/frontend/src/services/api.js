import axios from 'axios';

// In development, Vite proxy will forward /api requests to http://localhost:8000
// In production, you may need to set this to your actual API URL
const API_BASE_URL = import.meta.env.PROD 
  ? 'http://localhost:8000'  // Update this for production
  : '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const rippleAPI = {
  // Load LFP data
  load: async (params) => {
    const response = await api.post('/load', params);
    return response.data;
  },

  // Detect ripples
  detect: async (params) => {
    const response = await api.post('/detect', params);
    return response.data;
  },

  // Normalize ripples
  normalize: async (params) => {
    const response = await api.post('/normalize', params);
    return response.data;
  },

  // Reject ripples
  reject: async (params) => {
    const response = await api.post('/reject', params);
    return response.data;
  },

  // Get all events
  getEvents: async (jobId) => {
    const response = await api.get('/events', { params: { job_id: jobId } });
    return response.data;
  },

  // Get specific event data
  getEvent: async (jobId, k) => {
    const response = await api.get(`/event/${k}`, { params: { job_id: jobId } });
    return response.data;
  },
};

