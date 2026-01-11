import React, { useState } from 'react';

const Rejection = ({ onReject, loading, jobId }) => {
  const [threshold, setThreshold] = useState(3.0);

  const handleReject = () => {
    if (!jobId) {
      alert('Please normalize ripples first!');
      return;
    }

    onReject({
      job_id: jobId,
      strict_threshold: threshold,
    });
  };

  return (
    <div className="section-group">
      <div className="section-title">Stage 4: Rejection</div>
      
      <div className="form-row">
        <label>Reject thresh (z):</label>
        <input
          type="number"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          min={0.0}
          max={10.0}
          step={0.1}
        />
      </div>

      <button
        className="function-btn"
        onClick={handleReject}
        disabled={loading || !jobId}
      >
        Reject Ripples
      </button>
    </div>
  );
};

export default Rejection;

