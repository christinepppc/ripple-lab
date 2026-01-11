import React, { useState } from 'react';

const Visualization = ({ onVisualize, loading, jobId }) => {
  const [showRaw, setShowRaw] = useState(true);
  const [showBp, setShowBp] = useState(true);
  const [showSpec, setShowSpec] = useState(true);
  const [showWinSp, setShowWinSp] = useState(true);
  const [showActSp, setShowActSp] = useState(true);

  const handleVisualize = () => {
    if (!jobId) {
      alert('Please reject ripples first!');
      return;
    }

    onVisualize({
      showRaw,
      showBp,
      showSpec,
      showWinSp,
      showActSp,
    });
  };

  return (
    <div className="section-group">
      <div className="section-title">Stage 5: Visualization</div>
      
      <div className="checkbox-group">
        <div className="checkbox-item">
          <input
            type="checkbox"
            id="raw"
            checked={showRaw}
            onChange={(e) => setShowRaw(e.target.checked)}
          />
          <label htmlFor="raw">Raw LFP</label>
        </div>
        <div className="checkbox-item">
          <input
            type="checkbox"
            id="bp"
            checked={showBp}
            onChange={(e) => setShowBp(e.target.checked)}
          />
          <label htmlFor="bp">BP LFP</label>
        </div>
        <div className="checkbox-item">
          <input
            type="checkbox"
            id="spec"
            checked={showSpec}
            onChange={(e) => setShowSpec(e.target.checked)}
          />
          <label htmlFor="spec">Spectrogram (Analysis Window)</label>
        </div>
        <div className="checkbox-item">
          <input
            type="checkbox"
            id="winsp"
            checked={showWinSp}
            onChange={(e) => setShowWinSp(e.target.checked)}
          />
          <label htmlFor="winsp">Spectrum (Analysis Window)</label>
        </div>
        <div className="checkbox-item">
          <input
            type="checkbox"
            id="actsp"
            checked={showActSp}
            onChange={(e) => setShowActSp(e.target.checked)}
          />
          <label htmlFor="actsp">Spectrum (Actual Event Duration)</label>
        </div>
      </div>

      <button
        className="function-btn"
        onClick={handleVisualize}
        disabled={loading || !jobId}
      >
        Visualize Ripples
      </button>
    </div>
  );
};

export default Visualization;

