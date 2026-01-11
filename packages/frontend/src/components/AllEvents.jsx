import React, { useState, useEffect } from 'react';
import { rippleAPI } from '../services/api';

const AllEvents = ({ jobId, visualizationOptions }) => {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (jobId) {
      loadEvents();
    }
  }, [jobId]);

  const loadEvents = async () => {
    if (!jobId) return;
    
    setLoading(true);
    setError(null);
    try {
      const data = await rippleAPI.getEvents(jobId);
      setEvents(data.items || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleEvent = async (k, accepted) => {
    // This would need to be implemented on the backend
    // For now, just update local state
    setEvents(events.map(e => 
      e.k === k ? { ...e, accepted } : e
    ));
  };

  if (loading) {
    return <div className="loading">Loading events...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!jobId || events.length === 0) {
    return (
      <div className="loading">
        <p>Complete the analysis pipeline to see events</p>
      </div>
    );
  }

  const visibleColumns = [
    { key: 'raw', label: 'Raw LFP', visible: visualizationOptions?.showRaw },
    { key: 'bp', label: 'BP LFP', visible: visualizationOptions?.showBp },
    { key: 'spec', label: 'Spectrogram (Analysis Window)', visible: visualizationOptions?.showSpec },
    { key: 'winsp', label: 'Spectrum (Analysis Window)', visible: visualizationOptions?.showWinSp },
    { key: 'actsp', label: 'Spectrum (Actual Event Duration)', visible: visualizationOptions?.showActSp },
  ].filter(col => col.visible);

  const acceptedCount = events.filter(e => e.accepted).length;
  const rejectedCount = events.filter(e => !e.accepted).length;

  return (
    <div>
      <div className="status-message" style={{ marginBottom: '20px' }}>
        Events: {acceptedCount} accepted | {rejectedCount} rejected
      </div>

      <div className="events-grid" style={{ gridTemplateColumns: `auto auto repeat(${visibleColumns.length}, 1fr)` }}>
        {/* Header */}
        <div className="events-header">✓</div>
        <div className="events-header">Event Id</div>
        {visibleColumns.map(col => (
          <div key={col.key} className="events-header">{col.label}</div>
        ))}

        {/* Rows */}
        {events.map(event => (
          <React.Fragment key={event.k}>
            <div className="event-cell checkbox-cell">
              <input
                type="checkbox"
                checked={event.accepted}
                onChange={(e) => handleToggleEvent(event.k, e.target.checked)}
              />
            </div>
            <div className={`event-cell id-cell ${event.accepted ? 'accepted' : 'rejected'}`}>
              Ripple {String(event.k + 1).padStart(3, '0')}
            </div>
            {visibleColumns.map(col => (
              <div key={col.key} className="event-cell">
                <div style={{ width: '200px', height: '100px', background: '#f0f0f0', borderRadius: '5px' }}>
                  {/* Placeholder for event visualization */}
                  <p style={{ textAlign: 'center', paddingTop: '40px', color: '#999' }}>
                    {col.label}
                  </p>
                </div>
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default AllEvents;

