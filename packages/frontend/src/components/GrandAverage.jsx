import React from 'react';
import Plot from 'react-plotly.js';

const GrandAverage = ({ avgRes, specF, mode = 'Auto-Detection' }) => {
  if (!avgRes) {
    return (
      <div className="loading">
        <p>Visualize ripples to see grand average</p>
      </div>
    );
  }

  // Placeholder for grand average plots
  // You'll need to implement based on your backend data structure
  // This would typically show spectrograms, LFP traces, etc.
  const placeholderTrace = {
    x: Array.from({ length: 100 }, (_, i) => i * 0.01),
    y: Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.1) * Math.exp(-i * 0.05)),
    type: 'scatter',
    mode: 'lines',
    name: 'Grand Average',
    line: { color: '#d62728' },
  };

  const layout = {
    title: mode,
    xaxis: { title: 'Time (s)' },
    yaxis: { title: 'Amplitude' },
    height: 300,
    margin: { l: 50, r: 50, t: 50, b: 50 },
  };

  return (
    <div style={{ marginBottom: '30px' }}>
      <div className="plot-container">
        <Plot
          data={[placeholderTrace]}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default GrandAverage;

