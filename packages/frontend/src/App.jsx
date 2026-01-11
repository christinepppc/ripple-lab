import React, { useState } from 'react';
import './App.css';
import { rippleAPI } from './services/api';

// Components
import LandingPage from './components/LandingPage';
import SignalSelection from './components/SignalSelection';
import Detection from './components/Detection';
import TimeFrequencyDecomp from './components/TimeFrequencyDecomp';
import Rejection from './components/Rejection';
import Visualization from './components/Visualization';
import SaveResults from './components/SaveResults';
import ProcessingNotebook from './components/ProcessingNotebook';
import GrandAverage from './components/GrandAverage';
import AllEvents from './components/AllEvents';

function App() {
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('');
  const [showApp, setShowApp] = useState(false);

  // State
  const [jobId, setJobId] = useState(null);
  const [lfpData, setLfpData] = useState(null);
  const [detRes, setDetRes] = useState(null);
  const [normRes, setNormRes] = useState(null);
  const [rejRes, setRejRes] = useState(null);
  const [avgRes, setAvgRes] = useState(null);
  const [visualizationOptions, setVisualizationOptions] = useState(null);
  const [fs, setFs] = useState(1000);

  const handleLoad = async (params) => {
    setLoading(true);
    setError(null);
    setStatus('Loading LFP data...');
    
    try {
      const response = await rippleAPI.load(params);
      if (response.ok) {
        setJobId(response.job_id);
        setFs(params.fs);
        setLfpData({ shape: response.shape });
        setStatus(`Loaded LFP data: ${response.shape[0]} channels, ${response.shape[1]} samples`);
        
        // Clear previous results
        setDetRes(null);
        setNormRes(null);
        setRejRes(null);
        setAvgRes(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load LFP data');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  const handleDetect = async (params) => {
    setLoading(true);
    setError(null);
    setStatus('Detecting ripples...');
    
    try {
      const response = await rippleAPI.detect(params);
      if (response.ok) {
        setDetRes({ n_events: response.n_events });
        setStatus(`Detected ${response.n_events} ripples`);
      }
    } catch (err) {
      setError(err.message || 'Failed to detect ripples');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  const handleNormalize = async (params) => {
    setLoading(true);
    setError(null);
    setStatus('Normalizing ripples...');
    
    try {
      const response = await rippleAPI.normalize(params);
      if (response.ok) {
        setNormRes(response);
        setStatus('Normalization complete');
      }
    } catch (err) {
      setError(err.message || 'Failed to normalize ripples');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async (params) => {
    setLoading(true);
    setError(null);
    setStatus('Rejecting ripples...');
    
    try {
      const response = await rippleAPI.reject(params);
      if (response.ok) {
        setRejRes({
          passed: response.passed,
          total: response.total,
        });
        setStatus(`${response.passed}/${response.total} ripples passed rejection`);
      }
    } catch (err) {
      setError(err.message || 'Failed to reject ripples');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  const handleVisualize = async (options) => {
    setLoading(true);
    setError(null);
    setStatus('Visualizing ripples...');
    
    try {
      setVisualizationOptions(options);
      // In a real implementation, you might fetch event data here
      setStatus('Visualization complete');
      setActiveTab(2); // Switch to All Events tab
    } catch (err) {
      setError(err.message || 'Failed to visualize ripples');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = () => {
    setStatus('Saving results...');
    // Implement save functionality
    setTimeout(() => {
      setStatus('Results saved successfully');
    }, 1000);
  };

  const tabs = [
    { id: 0, label: 'Processing Notebook' },
    { id: 1, label: 'Grand Average' },
    { id: 2, label: 'All Events Analysis' },
  ];

  return (
    <div className="app-wrapper">
      {/* Landing Page - disappears after scroll */}
      <LandingPage onEnter={() => setShowApp(true)} />
      
      {/* Main Application - Shows when landing page is scrolled past */}
      <div className={`app-container ${showApp ? 'visible' : ''}`} id="main-app">
        {/* Sidebar */}
        <div className="sidebar-wrapper">
        <button
          className="sidebar-toggle"
          onClick={() => setSidebarVisible(!sidebarVisible)}
        >
          {sidebarVisible ? '«' : '»'}
        </button>
        
        {sidebarVisible && (
          <div className="sidebar">
            {error && <div className="error">{error}</div>}
            {status && <div className="status-message">{status}</div>}
            
            <SignalSelection onLoad={handleLoad} loading={loading} />
            <Detection onDetect={handleDetect} loading={loading} jobId={jobId} />
            <TimeFrequencyDecomp onNormalize={handleNormalize} loading={loading} jobId={jobId} />
            <Rejection onReject={handleReject} loading={loading} jobId={jobId} />
            <Visualization onVisualize={handleVisualize} loading={loading} jobId={jobId} />
            <SaveResults onSave={handleSave} loading={loading} jobId={jobId} status={status} />
          </div>
        )}
        </div>

        {/* Main Content */}
        <div className="main-content">
        <div className="tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="tab-content">
          {activeTab === 0 && (
            <ProcessingNotebook lfpData={lfpData} detRes={detRes} fs={fs} />
          )}
          {activeTab === 1 && (
            <div>
              <GrandAverage avgRes={avgRes} mode="Auto-Detection" />
              <GrandAverage avgRes={avgRes} mode="User Manual Selection" />
            </div>
          )}
          {activeTab === 2 && (
            <AllEvents jobId={jobId} visualizationOptions={visualizationOptions} />
          )}
        </div>
      </div>
      </div>
    </div>
  );
}

export default App;

