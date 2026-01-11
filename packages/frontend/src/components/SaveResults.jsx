import React from 'react';

const SaveResults = ({ onSave, loading, jobId, status }) => {
  const handleSave = () => {
    if (!jobId) {
      alert('Please visualize ripples first!');
      return;
    }
    onSave();
  };

  return (
    <div className="section-group">
      <div className="section-title">Stage 6: Save User Selected Results</div>
      
      {status && (
        <div className="status-message">{status}</div>
      )}

      <button
        className="function-btn"
        onClick={handleSave}
        disabled={loading || !jobId}
      >
        Save Ripples
      </button>
    </div>
  );
};

export default SaveResults;

