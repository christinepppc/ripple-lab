import React, { useState } from 'react';

const TimeFrequencyDecomp = ({ onNormalize, loading, jobId }) => {
  const [freqLower, setFreqLower] = useState(2);
  const [freqUpper, setFreqUpper] = useState(200);
  const [windowTime, setWindowTime] = useState(0.060);
  const [stepTime, setStepTime] = useState(0.001);
  const [timeBandwidth, setTimeBandwidth] = useState(1.2);
  const [tapers, setTapers] = useState(2);
  const [pad, setPad] = useState(20);

  const handleNormalize = () => {
    if (!jobId) {
      alert('Please detect ripples first!');
      return;
    }

    if (freqUpper <= freqLower) {
      alert('Upper frequency must be > lower frequency.');
      return;
    }

    onNormalize({
      job_id: jobId,
      fmin: freqLower,
      fmax: freqUpper,
      win_length: windowTime,
      step: stepTime,
      nw: timeBandwidth,
      tapers: tapers,
      tfspec_pad: pad,
    });
  };

  return (
    <div className="section-group">
      <div className="section-title">Stage 3: Time-Frequency Decomposition</div>
      
      <div className="form-row">
        <label>Frequency Analysis Range:</label>
        <div className="range-inputs">
          <input
            type="number"
            value={freqLower}
            onChange={(e) => setFreqLower(parseInt(e.target.value))}
            min={0}
            max={150}
            step={1}
          />
          <span>Hz</span>
          <input
            type="number"
            value={freqUpper}
            onChange={(e) => setFreqUpper(parseInt(e.target.value))}
            min={151}
            max={500}
            step={1}
          />
          <span>Hz</span>
        </div>
      </div>

      <div className="form-row">
        <label>Window Parameters [Window Time, Step Size]:</label>
        <div className="range-inputs">
          <input
            type="number"
            value={windowTime}
            onChange={(e) => setWindowTime(parseFloat(e.target.value))}
            min={0.001}
            max={1.000}
            step={0.010}
          />
          <span>s</span>
          <input
            type="number"
            value={stepTime}
            onChange={(e) => setStepTime(parseFloat(e.target.value))}
            min={0.001}
            max={1.000}
            step={0.001}
          />
          <span>s</span>
        </div>
      </div>

      <div className="form-row">
        <label>Time-Bandwidth Product (NW):</label>
        <input
          type="number"
          value={timeBandwidth}
          onChange={(e) => setTimeBandwidth(parseFloat(e.target.value))}
          min={0}
          max={7}
          step={0.1}
        />
      </div>

      <div className="form-row">
        <label>Number of Tapers (K = 2NW - 1):</label>
        <input
          type="number"
          value={tapers}
          onChange={(e) => setTapers(parseInt(e.target.value))}
          min={0}
          max={7}
          step={1}
        />
      </div>

      <div className="form-row">
        <label>Pad Signal:</label>
        <input
          type="number"
          value={pad}
          onChange={(e) => setPad(parseInt(e.target.value))}
          min={0}
          max={200}
          step={1}
        />
      </div>

      <button
        className="function-btn"
        onClick={handleNormalize}
        disabled={loading || !jobId}
      >
        Normalize Ripples
      </button>
    </div>
  );
};

export default TimeFrequencyDecomp;

