# Bipolar Processing and Ripple Analysis

This directory contains scripts for bipolar re-referencing and ripple analysis of neural data.

## Scripts Overview

### Core Scripts (Keep These)

1. **`complete_bipolar_workflow.py`** - **MAIN SCRIPT** - Unified workflow for all bipolar processing and ripple analysis
2. **`process_bipolar.py`** - Single trial bipolar processing (used by main workflow)
3. **`batch_process_bipolar.py`** - Batch processing for multiple trials (used by main workflow)
4. **`analyze_bipolar_matrix_simple.py`** - Ripple analysis on processed data (used by main workflow)

## Usage

### Complete Workflow (Recommended)

The `complete_bipolar_workflow.py` script provides three main commands:

#### 1. Single Trial Processing
```bash
python scripts/bipolar/complete_bipolar_workflow.py single /path/to/trial/dir --output /path/to/output
```

#### 2. Batch Processing
```bash
python scripts/bipolar/complete_bipolar_workflow.py batch --base-dir /path/to/data --sessions 134,135,136 --trials 001,002,003 --output /path/to/output
```

#### 3. Ripple Analysis Only
```bash
python scripts/bipolar/complete_bipolar_workflow.py analyze --matrix /path/to/bipolar_lfp_matrix.mat --output /path/to/output
```

### Individual Scripts (Advanced Usage)

If you need more control, you can use the individual scripts:

#### Bipolar Processing
```bash
# Single trial
python scripts/bipolar/process_bipolar.py /path/to/trial/dir --output /path/to/output --bad-channels 1,2,3

# Batch processing
python scripts/bipolar/batch_process_bipolar.py --base-dir /path/to/data --sessions 134,135,136 --trials 001,002,003 --output /path/to/output
```

#### Ripple Analysis
```bash
python scripts/bipolar/analyze_bipolar_matrix_simple.py
```

## Workflow Steps

1. **Bipolar Re-referencing**: Convert monopolar channels to bipolar pairs, excluding white matter channels
2. **Ripple Detection**: Detect ripples in the bipolar-referenced data
3. **Ripple Normalization**: Normalize ripple characteristics
4. **Ripple Rejection**: Apply quality control to reject poor-quality ripples
5. **Statistics**: Generate comprehensive statistics and reports

## Key Features

- **White Matter Exclusion**: Automatically excludes 134 white matter channels
- **Bad Channel Handling**: Supports manual specification of additional bad channels
- **Batch Processing**: Process multiple sessions and trials efficiently
- **Comprehensive Statistics**: Detailed ripple detection and quality metrics
- **Error Handling**: Robust error handling with detailed logging

## Output Files

- `bipolar_lfp_matrix.mat`: Processed bipolar LFP data
- `processing_summary.json`: Processing metadata and statistics
- `ripple_analysis_*.json`: Detailed ripple analysis results
- `batch_processing_summary.json`: Batch processing summary

## Configuration

The scripts use these default parameters:

### Bipolar Processing
- Excludes 134 white matter channels automatically
- Creates bipolar pairs from remaining channels
- Saves results as MATLAB .mat files

### Ripple Analysis
- Frequency band: 100-140 Hz
- Z-score threshold: 2.5 (detection), 9.0 (outlier)
- Minimum duration: 30 ms
- Merge duration: 10 ms
- Rejection threshold: 3.0

## Dependencies

- `ripple_core`: Core analysis functions
- `numpy`, `scipy`: Numerical processing
- `matplotlib`: Visualization (optional)
- `colorcet`: Colormap support

## Notes

- All scripts automatically handle white matter channel exclusion
- Processing is optimized for 1000 Hz sampling rate
- Results are saved in both MATLAB and JSON formats for compatibility
- The workflow is designed to be robust and handle missing data gracefully
