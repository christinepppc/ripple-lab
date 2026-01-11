import React, { useState } from 'react';

const Detection = ({ onDetect, loading, jobId }) => {
  const [bpTaps, setBpTaps] = useState(550);
  const [bpLow, setBpLow] = useState(100);
  const [bpHigh, setBpHigh] = useState(140);
  const [rmsLength, setRmsLength] = useState(20.0);
  const [outlier, setOutlier] = useState(9.0);
  const [lowerBound, setLowerBound] = useState(2.5);
  const [minDur, setMinDur] = useState(30);
  const [mergeDur, setMergeDur] = useState(10);
  const [visualSize, setVisualSize] = useState(200);

  const handleDetect = () => {
    if (!jobId) {
      alert('Please load LFP data first!');
      return;
    }

    onDetect({
      job_id: jobId,
      rp_low: bpLow,
      rp_high: bpHigh,
      order: bpTaps,
      window_ms: rmsLength,
      z_low: lowerBound,
      z_outlier: outlier,
      min_dur_ms: minDur,
      merge_dur_ms: mergeDur,
      epoch_ms: visualSize,
    });
  };

  return (
    <div className="section-group">
      <div className="section-title">Stage 2: Detection</div>
      
      <div className="form-row">
        <label>Filter Taps:</label>
        <input
          type="number"
          value={bpTaps}
          onChange={(e) => setBpTaps(parseInt(e.target.value))}
          min={1}
          max={1000}
        />
      </div>

      <div className="form-row">
        <label>BP Range (Hz):</label>
        <div className="range-inputs">
          <input
            type="number"
            value={bpLow}
            onChange={(e) => setBpLow(parseInt(e.target.value))}
            min={1}
            max={1000}
          />
          <span>–</span>
          <input
            type="number"
            value={bpHigh}
            onChange={(e) => setBpHigh(parseInt(e.target.value))}
            min={1}
            max={1000}
          />
        </div>
      </div>

      <div className="form-row">
        <label>RMS Length (ms):</label>
        <input
          type="number"
          value={rmsLength}
          onChange={(e) => setRmsLength(parseFloat(e.target.value))}
          min={1.0}
          max={1000.0}
          step={0.1}
        />
      </div>

      <div className="form-row">
        <label>RMS Remove Outlier:</label>
        <input
          type="number"
          value={outlier}
          onChange={(e) => setOutlier(parseFloat(e.target.value))}
          min={1.0}
          max={20.0}
          step={0.5}
        />
      </div>

      <div className="form-row">
        <label>RMS Threshold (mu + z * sd):</label>
        <input
          type="number"
          value={lowerBound}
          onChange={(e) => setLowerBound(parseFloat(e.target.value))}
          min={1}
          max={10}
          step={0.5}
        />
      </div>

      <div className="form-row">
        <label>Minimum Duration (ms):</label>
        <input
          type="number"
          value={minDur}
          onChange={(e) => setMinDur(parseFloat(e.target.value))}
          min={1}
          max={100}
          step={1}
        />
      </div>

      <div className="form-row">
        <label>Merge Duration (ms):</label>
        <input
          type="number"
          value={mergeDur}
          onChange={(e) => setMergeDur(parseFloat(e.target.value))}
          min={1}
          max={100}
          step={1}
        />
      </div>

      <div className="form-row">
        <label>Ripple Analysis Window (± ms, Ripple Peak Centered):</label>
        <input
          type="number"
          value={visualSize}
          onChange={(e) => setVisualSize(parseInt(e.target.value))}
          min={70}
          max={1000}
          step={1}
        />
      </div>

      <button
        className="function-btn"
        onClick={handleDetect}
        disabled={loading || !jobId}
      >
        Detect Ripples
      </button>
    </div>
  );
};

export default Detection;

