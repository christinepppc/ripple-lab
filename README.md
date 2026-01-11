# Ripple Lab

Comprehensive toolkit for ripple detection, analysis, and visualization in neural data.

## Project Structure

```
ripple-lab/
├── packages/
│   └── ripple_core/          # Core library (reusable functions)
│       ├── ripple_core/
│       │   ├── analyze.py    # Analysis algorithms
│       │   ├── load.py       # Data I/O functions
│       │   ├── labels.py     # Channel labeling & region categorization
│       │   ├── visualize.py  # Plotting utilities
│       │   └── signal/
│       │       └── bipolar.py  # Signal processing & ripple detection
│       └── setup.py          # Package installation
│
└── scripts/                  # Command-line workflows
    ├── preprocessing/        # Data loading & re-referencing
    ├── detection/            # Ripple detection workflows
    └── analysis/             # Analysis & visualization tools
```

---

## Installation

### 1. Install the core library

```bash
pip install -e packages/ripple_core/
```

This installs `ripple_core` in development mode, so you can:
- Import it from anywhere: `from ripple_core.load import load_electrodes`
- Make changes and they take effect immediately
- No more `sys.path.insert()` hacks!

### 2. Verify installation

```bash
python -c "import ripple_core; print(ripple_core.__version__)"
# Should print: 0.1.0
```

---

## Quick Start

### Complete Analysis Pipeline

```bash
# 1. Load raw LFP data
python scripts/preprocessing/load_sessions.py \
    --session_idx 45 --trial 1 --channels 1-220

# 2. Bipolar re-referencing
python scripts/preprocessing/process_bipolar_trial.py \
    /path/to/session46/trial001

# 3. Detect ripples
python scripts/detection/detect_ripples_bipolar.py \
    /path/to/session46/trial001_bipolar --z-low 3.0

# 4. Create channel labels
python scripts/analysis/create_channel_labels.py \
    --trial_bipolar_dir /path/to/trial001_bipolar

# 5. Analyze synchrony
python scripts/analysis/run_synchrony_analysis.py \
    --trial_dir /path/to/trial001_bipolar \
    --session_name session46_trial001
```

---

## Core Library (`packages/ripple_core/`)

The `ripple_core` package provides reusable functions for:

### Data Loading (`ripple_core.load`)
- `load_movie_database()`: Parse session metadata
- `load_electrodes()`: Load raw LFP from `.dat` files
- `load_all_channels()`: Load all channels for a session/trial

### Channel Labeling (`ripple_core.labels`)
- `CHANNEL_LABELS`: Mapping of channels 1-220 to anatomical labels
- `get_region_category()`: Categorize labels into broad regions
- `categorize_bipolar_pair()`: Classify bipolar pairs by region
- `create_bipolar_channel_labels()`: Generate full bipolar label mapping

### Signal Processing (`ripple_core.signal.bipolar`)
- `make_bipolar_pairs_from_grid()`: Generate bipolar pairs from electrode grid
- `process_bipolar_referencing()`: Apply bipolar re-referencing to a trial
- `detect_ripples_on_bipolar_channels()`: Detect ripples using bandpower method

### Analysis (`ripple_core.analyze`)
- Spectral analysis functions
- Ripple clustering and normalization
- Statistical analysis utilities

### Visualization (`ripple_core.visualize`)
- Plotting utilities for ripples and spectrograms
- Multi-channel visualization tools

---

## Scripts Directory

See `scripts/README.md` for detailed documentation of all command-line tools.

**Quick reference:**
- **Preprocessing:** `load_sessions.py`, `process_bipolar_trial.py`
- **Detection:** `detect_ripples_bipolar.py`, `test_detection_visualization.py`
- **Analysis:** `create_channel_labels.py`, `run_synchrony_analysis.py`

---

## Key Features

### 1. Bipolar Re-referencing
- Automatically generates horizontal and vertical pairs
- Excludes white matter channels
- Stays within electrode banks (headstages)
- Preserves spatial organization

### 2. Ripple Detection
- Bandpass filtering (100-140 Hz)
- RMS envelope calculation
- Z-score thresholding (configurable)
- Duration-based cleanup (min 30ms)
- Epoch extraction around peaks

### 3. Channel Labeling
- 220-channel anatomical mapping
- Region categorization (prefrontal, parietal, motor, etc.)
- Smart bipolar pair classification:
  - `within_X`: Both channels in region X (or one is white matter)
  - `cross_X_Y`: Channels in different regions
  - `mixed_or_unknown`: Unclear or both white matter

### 4. Synchrony Analysis
- Pairwise co-occurrence vs. spatial distance
- Multiple statistical controls:
  - Circular shift null (rate-preserving)
  - Permutation test (distance shuffle)
  - Robustness to window size
- Region-specific analysis
- Rate-controlled metrics

---

## Data Organization

### Input Data Structure
```
/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/
└── 180212/                  # Session date
    ├── 001/                 # Trial number
    │   ├── rec001.Frontal_1.lfp.dat
    │   ├── rec001.Frontal_2.lfp.dat
    │   └── ...
    └── Movie_Database.m     # Session metadata
```

### Output Data Structure
```
/vol/brains/bd3/pesaranlab/.../Chen/
└── session46/
    ├── trial001/                   # Raw LFP
    │   └── chan###/sess46_trial001_chan###.mat
    │
    ├── trial001_bipolar/           # Bipolar + Ripples
    │   ├── b001/
    │   │   ├── lfp_b001.mat
    │   │   ├── pair_info.txt
    │   │   └── ripples_b001_zlow3.0.mat
    │   ├── pairs_used.mat
    │   ├── bipolar_channel_labels.csv
    │   ├── ripple_detection_summary_zlow3.0.mat
    │   └── synchrony_analysis/
    │       ├── synchrony_parietal/
    │       ├── synchrony_prefrontal/
    │       └── region_comparison_summary.csv
    └── channel_labels.csv          # Session-level labels
```

---

## Region Categories

Channels are categorized into these broad regions:

- **`prefrontal`**: Frontal gyri, anterior cingulate, orbital gyri
- **`motor`**: Precentral gyrus
- **`parietal`**: Parietal lobules, precuneus, postcentral, supramarginal
- **`basal_ganglia`**: Caudate, putamen, globus pallidus, nucleus accumbens
- **`mtl`**: Hippocampus, subiculum, entorhinal cortex
- **`amygdala`**: Amygdalar nuclei
- **`thalamus`**: Thalamic nuclei
- **`white_matter`**: White matter tracts
- **`other`**: Corpus callosum, internal capsule, optic tract, etc.
- **`unknown`**: Empty or unrecognized labels

---

## Development

### Testing Changes
```bash
# Make changes to ripple_core
# Changes take effect immediately (development install)
python -c "from ripple_core.labels import get_region_category; print(get_region_category('r_superior_parietal_lobule'))"
```

### Adding New Functions
1. Add to appropriate module in `packages/ripple_core/ripple_core/`
2. Import in `__init__.py` if needed
3. Document with docstrings
4. Create a script in `scripts/` if it's a workflow

---

## Authors

Pesaran Lab - Neural data analysis and ripple detection

---

## License

Internal research use - Pesaran Lab
