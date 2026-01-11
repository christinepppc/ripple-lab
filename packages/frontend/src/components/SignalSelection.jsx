import React, { useState } from 'react';

const SignalSelection = ({ onLoad, loading }) => {
  const [fs, setFs] = useState(1000);
  const [session, setSession] = useState(1);
  const [trial, setTrial] = useState(1);
  const [mode, setMode] = useState('Select Channel(s)');
  const [channel, setChannel] = useState(1);
  const [region, setRegion] = useState('(r) anterior amygdalar area');

  const regions = [
    '(r) anterior amygdalar area',
    '(r) anterior cingulate gyrus',
    '(r) caudate nucleus',
    '(r) central amygdalar nucleus',
    '(r) cerebral white matter',
    '(r) frontal white matter',
    '(r) fronto-orbital gyrus',
    '(r) genu of the corpus callosum',
    '(r) internal capsule',
    '(r) lateral globus pallidus',
    '(r) lateral orbital gyrus',
    '(r) medial orbital gyrus',
    '(r) middle frontal gyrus',
    '(r) nucleus accumbens',
    '(r) optic tract',
    '(r) postcentral gyrus',
    '(l) postcentral gyrus',
    '(r) posterior cingulate gyrus',
    '(r) precentral gyrus',
    '(r) precuneus',
    '(r) presubiculum',
    '(r) putamen',
    '(r) superior frontal gyrus',
    '(r) superior parietal lobule',
    '(r) supramarginal gyrus',
    '(r) thalamus'
  ];

  const handleLoad = () => {
    const params = {
      mode,
      session,
      trial,
      fs,
    };

    if (mode === 'Select Channel(s)') {
      params.channel = channel;
    } else if (mode === 'Select Region(s)') {
      params.region = region;
    }

    onLoad(params);
  };

  return (
    <div className="section-group">
      <div className="section-title">Stage 1: LFP Signal Selection</div>
      
      <div className="form-row">
        <label>Fs:</label>
        <input
          type="number"
          value={fs}
          onChange={(e) => setFs(parseInt(e.target.value))}
          min={800}
          max={1500}
        />
      </div>

      <div className="form-row">
        <label>Session:</label>
        <input
          type="number"
          value={session}
          onChange={(e) => setSession(parseInt(e.target.value))}
          min={1}
          max={151}
        />
      </div>

      <div className="form-row">
        <label>Trial:</label>
        <input
          type="number"
          value={trial}
          onChange={(e) => setTrial(parseInt(e.target.value))}
          min={1}
          max={10}
        />
      </div>

      <div className="form-row">
        <label>Channel Load Mode:</label>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option>Select Channel(s)</option>
          <option>Select Region(s)</option>
          <option>All Channels</option>
        </select>
      </div>

      {mode === 'Select Channel(s)' && (
        <div className="form-row">
          <label>Channel:</label>
          <input
            type="number"
            value={channel}
            onChange={(e) => setChannel(parseInt(e.target.value))}
            min={1}
            max={220}
          />
        </div>
      )}

      {mode === 'Select Region(s)' && (
        <div className="form-row">
          <label>Region:</label>
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            {regions.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
      )}

      <button
        className="function-btn"
        onClick={handleLoad}
        disabled={loading}
      >
        Load LFP
      </button>
    </div>
  );
};

export default SignalSelection;

