"""
Multivariate Granger Causality Analysis using Sparse VAR (Lasso)

This script implements:
- Stage 1: Preprocessing (bandpass filter + RMS computation via detection pipeline)
- Stage 2: Artifact removal (using rejection pipeline to mask rejected ripples)
- Stage 3: Sparse MVGC

Author: Generated for ripple-lab
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# Add ripple_core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "ripple_core"))
from ripple_core.analyze import detect_ripples, reject_ripples_fast


def load_lfp_data(bipolar_dir: Path) -> Tuple[np.ndarray, float]:
    """
    Load LFP data from lfp_b0xx.mat file.
    
    Args:
        bipolar_dir: Path to bipolar channel directory (e.g., b001/)
    
    Returns:
        (lfp_signal, sampling_rate)
    """
    lfp_file = bipolar_dir / f"lfp_{bipolar_dir.name}.mat"
    if not lfp_file.exists():
        raise FileNotFoundError(f"LFP file not found: {lfp_file}")
    
    data = sio.loadmat(str(lfp_file))
    
    # Try common variable names for LFP data
    lfp = None
    for key in ['lfp', 'data', 'signal', 'LFP', 'Data']:
        if key in data:
            lfp = data[key]
            break
    
    if lfp is None:
        # If not found, try to get the first non-metadata array
        keys = [k for k in data.keys() if not k.startswith('__')]
        if keys:
            lfp = data[keys[0]]
    
    if lfp is None:
        raise ValueError(f"Could not find LFP data in {lfp_file}")
    
    # Flatten if needed
    lfp = np.squeeze(lfp)
    if lfp.ndim > 1:
        lfp = lfp.flatten()
    lfp = lfp.astype(np.float64)
    
    # Get sampling rate
    fs = 1000.0  # Default
    for key in ['fs', 'Fs', 'sampling_rate', 'rate']:
        if key in data:
            fs_val = data[key]
            if isinstance(fs_val, np.ndarray):
                fs = float(fs_val.item())
            else:
                fs = float(fs_val)
            if not (np.isnan(fs) or fs <= 0):
                break
            else:
                fs = 1000.0
    
    return lfp, fs


def process_channel_with_rejection(
    lfp: np.ndarray,
    fs: float,
    rp_band: Tuple[int, int] = (100, 140),
    z_low: float = 3.0,
    strict_threshold: float = 3.0,
    window_ms: int = 20,
    order: int = 550,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run full detection and rejection pipeline on raw LFP.
    
    This function:
    1. Runs detect_ripples() which computes RMS envelope (env_rip)
    2. Runs reject_ripples_fast() to get rejection markers
    3. Returns the RMS envelope and rejection information
    
    Args:
        lfp: Raw LFP signal
        fs: Sampling rate
        rp_band: Ripple frequency band
        z_low: Z-score threshold for detection
        strict_threshold: Z-score threshold for rejection
        window_ms: RMS window size
        order: FIR filter order
    
    Returns:
        (rms_envelope, rejected_ripples_array, metadata_dict)
        - rms_envelope: The RMS envelope from detection (env_rip)
        - rejected_ripples_array: Array of shape (n_rejected, 2) with [start_idx, end_idx]
        - metadata_dict: Contains detection and rejection statistics
    """
    # Step 1: Detect ripples (this computes the RMS envelope)
    det_result = detect_ripples(
        lfp,
        fs=fs,
        rp_band=rp_band,
        order=order,
        window_ms=window_ms,
        z_low=z_low,
        z_outlier=9.0,
        min_dur_ms=30,
        merge_dur_ms=10,
        epoch_ms=200,
    )
    
    # The RMS envelope is det_result.env_rip
    rms_envelope = det_result.env_rip.copy()
    
    # Step 2: Run rejection pipeline
    if len(det_result.real_duration) == 0:
        # No ripples detected, nothing to reject
        return rms_envelope, np.array([]).reshape(0, 2), {
            'n_detected': 0,
            'n_rejected': 0,
            'n_passed': 0,
        }
    
    rej_result = reject_ripples_fast(
        lfp=lfp,
        bp_lfp=det_result.bp_lfp,
        fs=int(fs),
        real_duration=det_result.real_duration,
        peak_idx=det_result.peak_idx,
        env_rip=det_result.env_rip,
        mu=float(det_result.mu),
        sd=float(det_result.sd),
        strict_threshold=strict_threshold,
        rp_band=rp_band,
        min_duration_ms=30.0,
        max_duration_ms=150.0,
        min_sharpness=0.5,
        bandpower_zscore_min=3.0,
    )
    
    # rej_result.markers is boolean array: True if rejected
    # Get rejected ripple timestamps
    rejected_indices = np.where(rej_result.markers)[0]
    n_rejected = len(rejected_indices)
    n_passed = len(rej_result.pass_idx)
    n_detected = len(det_result.real_duration)
    
    rejected_ripples = det_result.real_duration[rejected_indices] if n_rejected > 0 else np.array([]).reshape(0, 2)
    
    metadata = {
        'n_detected': n_detected,
        'n_rejected': n_rejected,
        'n_passed': n_passed,
        'rejection_reasons': rej_result.reasons if hasattr(rej_result, 'reasons') else [],
    }
    
    return rms_envelope, rejected_ripples, metadata


def mask_rejected_ripples(
    rms_envelope: np.ndarray, 
    rejected_ripples: np.ndarray, 
    fs: float, 
    window_ms: float = 20.0
) -> np.ndarray:
    """
    Mask rejected ripple segments by setting RMS values to channel mean.
    
    This function correctly locates rejected ripples from the rejection pipeline
    and wipes out their RMS values by setting them to the channel mean.
    
    Args:
        rms_envelope: RMS envelope to mask
        rejected_ripples: Array of shape (n_rejected, 2) with [start_idx, end_idx]
        fs: Sampling rate
        window_ms: RMS window size (for determining overlap)
    
    Returns:
        Masked RMS envelope
    """
    if rejected_ripples is None or len(rejected_ripples) == 0:
        return rms_envelope
    
    masked_rms = rms_envelope.copy()
    channel_mean = np.mean(rms_envelope)
    
    window_samples = int(window_ms * fs / 1000.0)
    
    for start_idx, end_idx in rejected_ripples:
        # Convert to integers
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        
        # Expand by half window on each side to account for RMS window overlap
        expanded_start = max(0, start_idx - window_samples // 2)
        expanded_end = min(len(masked_rms), end_idx + window_samples // 2)
        
        # Set RMS values in this range to channel mean
        masked_rms[expanded_start:expanded_end] = channel_mean
    
    return masked_rms


def fit_sparse_var(
    rms_matrix: np.ndarray, 
    fs: float,
    max_lag_ms: float = 50.0,
    alpha: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Fit Sparse Vector Autoregressive (VAR) model using Lasso regression.
    
    This implements a sparse VAR model where for each channel i, we regress 
    future values on past values of all channels:
    X_i(t) = sum_{j=1}^{n_channels} sum_{l=1}^{max_lag} A_{ij}(l) * X_j(t-l) + epsilon_i(t)
    
    The model is fit separately for each channel using L1-regularized regression (Lasso),
    which encourages sparsity in the connectivity matrix.
    
    FIXES APPLIED:
    1. Standardizes X (and y) before Lasso to handle scale differences
    2. Zeros out diagonal (self-connections) after computation
    
    Args:
        rms_matrix: RMS matrix of shape (n_samples, n_channels)
        fs: Sampling rate
        max_lag_ms: Maximum lag in milliseconds (default: 50 ms)
        alpha: L1 regularization strength for Lasso (default: 0.1)
    
    Returns:
        Connectivity matrix of shape (n_channels, n_channels) where [i,j] 
        represents total causal strength from channel j to channel i
        (diagonal is zeroed out - no self-connections)
    """
    n_samples, n_channels = rms_matrix.shape
    
    # Convert max_lag from milliseconds to samples
    max_lag = int(np.round(max_lag_ms * fs / 1000.0))
    if max_lag < 1:
        max_lag = 1
    
    if logger:
        logger.info(f"  Using max_lag={max_lag} samples ({max_lag_ms} ms) with alpha={alpha}")
    print(f"  Using max_lag={max_lag} samples ({max_lag_ms} ms) with alpha={alpha}")
    
    # Create lagged features
    # For each time point t >= max_lag, we have:
    # - Target: X_i(t) for channel i
    # - Features: X_j(t-l) for all channels j and lags l=1..max_lag
    
    n_valid = n_samples - max_lag
    
    # Build feature matrix: [X_1(t-1), ..., X_78(t-1), X_1(t-2), ..., X_78(t-2), ...]
    X = np.zeros((n_valid, n_channels * max_lag))
    for lag in range(1, max_lag + 1):
        start_col = (lag - 1) * n_channels
        end_col = lag * n_channels
        X[:, start_col:end_col] = rms_matrix[max_lag - lag:n_samples - lag, :]
    
    # Build target matrix: [X_1(t), X_2(t), ..., X_78(t)] for t >= max_lag
    Y = rms_matrix[max_lag:, :]
    
    # CRITICAL FIX: Standardize X and Y before Lasso
    # Lasso is scale-sensitive, so we need to standardize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)
    
    # Fit Sparse VAR: Lasso for each channel
    # This is the standard approach for sparse VAR modeling
    connectivity = np.zeros((n_channels, n_channels))
    
    var_start = datetime.now()
    for i in range(n_channels):
        ch_start = datetime.now()
        y = Y_scaled[:, i]
        
        # Fit Lasso (L1-regularized regression) for this channel
        lasso = Lasso(alpha=alpha, max_iter=2000, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Extract coefficients and aggregate across lags for each source channel
        # Coefficients are organized as: [lag1_ch1, lag1_ch2, ..., lag1_chN, lag2_ch1, ...]
        coefs = lasso.coef_.reshape(max_lag, n_channels)
        
        # Sum absolute values across lags for each source channel
        # This gives total causal strength from each source channel
        connectivity[i, :] = np.sum(np.abs(coefs), axis=0)
        
        # Log progress every 10 channels
        if logger and (i + 1) % 10 == 0:
            elapsed = (datetime.now() - var_start).total_seconds()
            logger.info(f"  VAR progress: {i+1}/{n_channels} channels fitted in {elapsed:.1f}s")
            print(f"  VAR progress: {i+1}/{n_channels} channels fitted...", flush=True)
    
    # CRITICAL FIX: Zero out diagonal (self-connections)
    # VAR will always use itself heavily, so we exclude self-causality
    np.fill_diagonal(connectivity, 0.0)
    
    return connectivity


def plot_connectivity_matrix(connectivity: np.ndarray, output_file: Optional[Path] = None):
    """
    Plot connectivity matrix as heatmap.
    
    Args:
        connectivity: Connectivity matrix (n_channels, n_channels)
        output_file: Optional path to save figure
    """
    plt.figure(figsize=(12, 10))
    
    if HAS_SEABORN:
        sns.heatmap(connectivity, cmap='hot', cbar=True, 
                    xticklabels=False, yticklabels=False,
                    square=True, linewidths=0.1)
    else:
        # Use matplotlib's imshow as fallback
        plt.imshow(connectivity, cmap='hot', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    
    plt.title('Multivariate Granger Causality Connectivity Matrix\n(Entry [i,j] = Causal strength from Ch j → Ch i)')
    plt.xlabel('Source Channel (j)')
    plt.ylabel('Target Channel (i)')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved connectivity plot to {output_file}")
    else:
        plt.show()


def main(
    trial_dir: Path, 
    output_dir: Optional[Path] = None, 
    z_low: float = 3.0,
    strict_threshold: float = 3.0,
    max_lag_ms: float = 50.0,
    alpha: float = 0.1,
    rp_band: Tuple[int, int] = (100, 140),
    window_ms: int = 20,
    save_rms: bool = True,
):
    """
    Main pipeline for Multivariate Granger Causality analysis using Sparse VAR.
    
    Args:
        trial_dir: Path to trial directory (e.g., trial001_bipolar/)
        output_dir: Optional output directory for results
        z_low: Z-score threshold for ripple detection
        strict_threshold: Z-score threshold for rejection
        max_lag_ms: Maximum lag in milliseconds (default: 50 ms)
        alpha: L1 regularization strength for sparse VAR (default: 0.1)
        rp_band: Ripple frequency band (low, high) in Hz
        window_ms: RMS window size (milliseconds)
        save_rms: Whether to save RMS envelopes
    """
    # Set up output directory and logging
    if output_dir is None:
        output_dir = trial_dir / "derived" / "mvgc"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to both file and console
    log_file = output_dir / "mvgc_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting MVGC analysis for {trial_dir}")
    logger.info(f"Parameters: z_low={z_low}, strict_threshold={strict_threshold}, "
                f"max_lag_ms={max_lag_ms}, alpha={alpha}")
    start_time = datetime.now()
    
    print(f"Starting MVGC analysis for {trial_dir}")
    print(f"Log file: {log_file}")
    
    # Get all bipolar channel directories
    bipolar_dirs = sorted([d for d in trial_dir.glob("b[0-9][0-9][0-9]") if d.is_dir()])
    
    if len(bipolar_dirs) == 0:
        raise ValueError(f"No bipolar channel directories found in {trial_dir}")
    
    logger.info(f"Found {len(bipolar_dirs)} channels")
    print(f"Found {len(bipolar_dirs)} channels")
    
    # Stage 1: Preprocessing + Detection + Rejection
    logger.info("=== Stage 1: Detection & RMS Computation ===")
    logger.info("=== Stage 2: Artifact Removal (Rejection) ===")
    print("\n=== Stage 1: Detection & RMS Computation ===")
    print("=== Stage 2: Artifact Removal (Rejection) ===")
    stage1_start = datetime.now()
    rms_envelopes = []
    rejected_counts = []
    fs_global = None
    
    for i, b_dir in enumerate(bipolar_dirs):
        try:
            ch_start = datetime.now()
            logger.info(f"Processing {b_dir.name} ({i+1}/{len(bipolar_dirs)})...")
            print(f"Processing {b_dir.name} ({i+1}/{len(bipolar_dirs)})...", end=' ', flush=True)
            
            # Load LFP
            lfp, fs = load_lfp_data(b_dir)
            if fs_global is None:
                fs_global = fs
            
            # Run detection and rejection pipeline
            rms, rejected, metadata = process_channel_with_rejection(
                lfp,
                fs=fs,
                rp_band=rp_band,
                z_low=z_low,
                strict_threshold=strict_threshold,
                window_ms=window_ms,
            )
            
            # Mask rejected ripples
            masked_rms = mask_rejected_ripples(rms, rejected, fs, window_ms=window_ms)
            rms_envelopes.append(masked_rms)
            
            rejected_counts.append(metadata['n_rejected'])
            ch_time = (datetime.now() - ch_start).total_seconds()
            logger.info(f"  {b_dir.name}: detected={metadata['n_detected']}, "
                       f"rejected={metadata['n_rejected']}, passed={metadata['n_passed']}, "
                       f"time={ch_time:.2f}s")
            print(f"✓ (detected: {metadata['n_detected']}, rejected: {metadata['n_rejected']}, passed: {metadata['n_passed']})")
            
            # Log progress every 10 channels
            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - stage1_start).total_seconds()
                logger.info(f"Progress: {i+1}/{len(bipolar_dirs)} channels processed in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Error processing {b_dir.name}: {e}", exc_info=True)
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            # Use zeros as placeholder
            if len(rms_envelopes) > 0:
                rms_envelopes.append(np.zeros_like(rms_envelopes[0]))
            else:
                raise
    
    stage1_time = (datetime.now() - stage1_start).total_seconds()
    logger.info(f"Stage 1 & 2 completed in {stage1_time:.1f}s")
    
    # Align lengths (take minimum)
    min_length = min(len(rms) for rms in rms_envelopes)
    rms_matrix = np.array([rms[:min_length] for rms in rms_envelopes]).T
    logger.info(f"RMS matrix shape: {rms_matrix.shape}")
    logger.info(f"Total rejected ripples across channels: {sum(rejected_counts)}")
    print(f"\nRMS matrix shape: {rms_matrix.shape}")
    print(f"Total rejected ripples across channels: {sum(rejected_counts)}")
    
    # Save RMS data if requested
    if save_rms:
        rms_file = output_dir / "rms_envelopes.npy"
        np.save(rms_file, rms_matrix)
        logger.info(f"Saved RMS envelopes to {rms_file}")
        print(f"Saved RMS envelopes to {rms_file}")
    
    # Stage 3: Sparse MVGC
    logger.info("=== Stage 3: Sparse Vector Autoregression (VAR) ===")
    logger.info(f"Fitting sparse VAR model with max_lag={max_lag_ms} ms, alpha={alpha}...")
    print("\n=== Stage 3: Sparse Vector Autoregression (VAR) ===")
    print(f"Fitting sparse VAR model with max_lag={max_lag_ms} ms, alpha={alpha}...")
    stage3_start = datetime.now()
    
    connectivity = fit_sparse_var(
        rms_matrix,
        fs=fs_global,
        max_lag_ms=max_lag_ms,
        alpha=alpha,
        logger=logger,
    )
    
    stage3_time = (datetime.now() - stage3_start).total_seconds()
    logger.info(f"Stage 3 completed in {stage3_time:.1f}s")
    
    logger.info(f"Connectivity matrix shape: {connectivity.shape}")
    logger.info(f"Non-zero connections (total): {np.count_nonzero(connectivity)}")
    logger.info(f"Max connection strength: {np.max(connectivity):.4f}")
    logger.info(f"Mean connection strength: {np.mean(connectivity):.4f}")
    print(f"Connectivity matrix shape: {connectivity.shape}")
    print(f"Non-zero connections (total): {np.count_nonzero(connectivity)}")
    print(f"  (Note: diagonal is zeroed - no self-connections)")
    print(f"Max connection strength: {np.max(connectivity):.4f}")
    print(f"Mean connection strength: {np.mean(connectivity):.4f}")
    
    if np.count_nonzero(connectivity) > 0:
        logger.info(f"Cross-channel connectivity: {np.count_nonzero(connectivity)} non-zero")
        print(f"\nCross-channel connectivity:")
        print(f"  Non-zero: {np.count_nonzero(connectivity)}")
        print(f"  Max: {np.max(connectivity):.4f}")
        print(f"  Mean (non-zero): {np.mean(connectivity[connectivity > 0]):.4f}")
        
        # Show top 10 cross-channel connections
        flat = connectivity.flatten()
        top_indices = np.argsort(flat)[-10:][::-1]
        logger.info("Top 10 cross-channel connections:")
        print(f"\n  Top 10 cross-channel connections:")
        for idx in top_indices:
            if flat[idx] > 0:
                i, j = np.unravel_index(idx, connectivity.shape)
                logger.info(f"    Ch {j:02d} -> Ch {i:02d}: {connectivity[i,j]:.4f}")
                print(f"    Ch {j:02d} -> Ch {i:02d}: {connectivity[i,j]:.4f}")
    
    # Save results
    
    # Save connectivity matrix
    connectivity_file = output_dir / "connectivity_matrix.npy"
    np.save(connectivity_file, connectivity)
    logger.info(f"Saved connectivity matrix to {connectivity_file}")
    print(f"\nSaved connectivity matrix to {connectivity_file}")
    
    # Save as MATLAB file
    mat_file = output_dir / "connectivity_matrix.mat"
    sio.savemat(str(mat_file), {
        'connectivity': connectivity,
        'channels': [d.name for d in bipolar_dirs],
        'max_lag_ms': max_lag_ms,
        'max_lag_samples': int(np.round(max_lag_ms * fs_global / 1000.0)),
        'alpha': alpha,
        'fs': fs_global,
        'z_low': z_low,
        'strict_threshold': strict_threshold,
    })
    logger.info(f"Saved MATLAB file to {mat_file}")
    print(f"Saved MATLAB file to {mat_file}")
    
    # Plot
    plot_file = output_dir / "connectivity_heatmap.png"
    plot_connectivity_matrix(connectivity, output_file=plot_file)
    logger.info(f"Saved connectivity plot to {plot_file}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"=== Analysis Complete ===")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("\n=== Analysis Complete ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Log file: {log_file}")
    return connectivity, rms_matrix


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multivariate Granger Causality Analysis using Sparse VAR (Lasso)"
    )
    parser.add_argument("trial_dir", type=str, 
                       help="Path to trial directory (e.g., trial001_bipolar/)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: trial_dir/derived/mvgc)")
    parser.add_argument("--z-low", type=float, default=3.0,
                       help="Z-score threshold for detection (default: 3.0)")
    parser.add_argument("--strict-threshold", type=float, default=3.0,
                       help="Z-score threshold for rejection (default: 3.0)")
    parser.add_argument("--max-lag-ms", type=float, default=50.0,
                       help="Maximum lag in milliseconds (default: 50 ms)")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="L1 regularization strength for sparse VAR (default: 0.1)")
    parser.add_argument("--low-freq", type=float, default=100.0,
                       help="Low cutoff frequency for bandpass (default: 100 Hz)")
    parser.add_argument("--high-freq", type=float, default=140.0,
                       help="High cutoff frequency for bandpass (default: 140 Hz)")
    parser.add_argument("--window-ms", type=int, default=20,
                       help="RMS window size in milliseconds (default: 20 ms)")
    parser.add_argument("--no-save-rms", action="store_true",
                       help="Don't save RMS envelopes")
    
    args = parser.parse_args()
    
    trial_dir = Path(args.trial_dir)
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    main(
        trial_dir=trial_dir,
        output_dir=output_dir,
        z_low=args.z_low,
        strict_threshold=args.strict_threshold,
        max_lag_ms=args.max_lag_ms,
        alpha=args.alpha,
        rp_band=(args.low_freq, args.high_freq),
        window_ms=args.window_ms,
        save_rms=not args.no_save_rms,
    )
