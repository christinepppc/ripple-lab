"""
Visualization functions for re-referenced neural data (bipolar and CAR).

This module provides comprehensive visualization functions that mirror the MATLAB
visualizeRipples.m functionality for both bipolar and CAR re-referenced signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.io as sio
from scipy.signal import welch, spectrogram
from scipy.signal.windows import dpss

from .analyze import detect_ripples
from .multitaper_spectrogram_python import multitaper_spectrogram


def _load_ripple_data(ripple_file: Path) -> Dict:
    """Load ripple detection results from .mat file."""
    data = sio.loadmat(ripple_file, squeeze_me=True)
    return {
        'peak_idx': data.get('peak_idx', np.array([])),
        'real_duration': data.get('real_duration', np.array([])),
        'mu': data.get('mu', 0.0),
        'sd': data.get('sd', 1.0),
        'fs': data.get('fs', 1000.0)
    }


def _extract_ripple_windows(lfp: np.ndarray, peak_idx: np.ndarray, fs: float, 
                           window_ms: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ripple windows around peak indices.
    
    Args:
        lfp: LFP signal
        peak_idx: Peak indices
        fs: Sampling frequency
        window_ms: Window size in milliseconds
        
    Returns:
        Tuple of (windowed_lfp, time_axis)
    """
    if len(peak_idx) == 0:
        return np.array([]), np.array([])
    
    window_samples = int(window_ms * fs / 1000)
    half_window = window_samples // 2
    
    windows = []
    for peak in peak_idx:
        start = max(0, int(peak) - half_window)
        end = min(len(lfp), int(peak) + half_window)
        
        if end - start == window_samples:
            windows.append(lfp[start:end])
        else:
            # Pad with zeros if at edge
            window = np.zeros(window_samples)
            actual_start = max(0, half_window - int(peak))
            actual_end = actual_start + (end - start)
            window[actual_start:actual_end] = lfp[start:end]
            windows.append(window)
    
    if len(windows) == 0:
        return np.array([]), np.array([])
    
    windowed_lfp = np.array(windows).T  # Shape: (time_points, n_ripples)
    time_axis = np.linspace(-window_ms, window_ms, window_samples)
    
    return windowed_lfp, time_axis


def _compute_spectrograms(windowed_lfp: np.ndarray, fs: float, 
                         fmin: float = 2, fmax: float = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrograms for windowed LFP data.
    
    Args:
        windowed_lfp: Windowed LFP data (time_points, n_ripples)
        fs: Sampling frequency
        fmin, fmax: Frequency range
        
    Returns:
        Tuple of (spectrograms, freq_axis, normalized_spectrograms)
    """
    if windowed_lfp.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    n_ripples = windowed_lfp.shape[1]
    spectrograms = []
    normalized_spectrograms = []
    
    for i in range(n_ripples):
        signal = windowed_lfp[:, i]
        
        # Compute multitaper spectrogram
        try:
            S, f, t = multitaper_spectrogram(
                signal, fs=fs, 
                frequency_range=[fmin, fmax], 
                window_params=[0.1, 0.01],
                plot_on=False,
                verbose=False
            )
            
            # Normalize spectrogram
            S_norm = (S - np.mean(S)) / np.std(S)
            
            spectrograms.append(S)
            normalized_spectrograms.append(S_norm)
            
        except Exception as e:
            print(f"Warning: Could not compute spectrogram for ripple {i}: {e}")
            # Create dummy spectrogram
            S = np.zeros((len(f), len(t)))
            spectrograms.append(S)
            normalized_spectrograms.append(S)
    
    if len(spectrograms) == 0:
        return np.array([]), np.array([]), np.array([])
    
    spectrograms = np.array(spectrograms)  # Shape: (n_ripples, freq, time)
    normalized_spectrograms = np.array(normalized_spectrograms)
    
    return spectrograms, f, normalized_spectrograms


def _compute_spectra(windowed_lfp: np.ndarray, fs: float, 
                    fmin: float = 2, fmax: float = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectra for windowed LFP data.
    
    Args:
        windowed_lfp: Windowed LFP data (time_points, n_ripples)
        fs: Sampling frequency
        fmin, fmax: Frequency range
        
    Returns:
        Tuple of (spectra, freq_axis)
    """
    if windowed_lfp.size == 0:
        return np.array([]), np.array([])
    
    n_ripples = windowed_lfp.shape[1]
    spectra = []
    
    for i in range(n_ripples):
        signal = windowed_lfp[:, i]
        
        # Compute power spectral density
        f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)//4))
        
        # Filter to frequency range
        mask = (f >= fmin) & (f <= fmax)
        f_filtered = f[mask]
        Pxx_filtered = Pxx[mask]
        
        spectra.append(Pxx_filtered)
    
    if len(spectra) == 0:
        return np.array([]), np.array([])
    
    spectra = np.array(spectra)  # Shape: (n_ripples, freq)
    freq_axis = f_filtered
    
    return spectra, freq_axis


def visualize_bipolar_ripples(bipolar_dir: str, output_dir: str, 
                             max_channels: int = 10, window_ms: int = 200):
    """
    Create comprehensive ripple visualizations for bipolar re-referenced data.
    
    Args:
        bipolar_dir: Directory containing bipolar_lfp_matrix.mat
        output_dir: Directory to save visualizations
        max_channels: Maximum number of channels to visualize
        window_ms: Window size for ripple extraction
    """
    print("=== Bipolar Ripple Visualization ===")
    
    bipolar_path = Path(bipolar_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load bipolar matrix
    matrix_file = bipolar_path / "bipolar_lfp_matrix.mat"
    if not matrix_file.exists():
        raise FileNotFoundError(f"Bipolar matrix not found: {matrix_file}")
    
    data = sio.loadmat(matrix_file, squeeze_me=True)
    lfp_matrix = data["lfp_matrix"]
    channel_ids = data["channel_ids"]
    fs = data.get("fs", 1000.0)
    
    if np.isnan(fs) or fs <= 0:
        fs = 1000.0
    
    print(f"Loaded bipolar matrix: {lfp_matrix.shape}")
    print(f"Channels: {len(channel_ids)}")
    print(f"Sampling frequency: {fs} Hz")
    
    # Process each channel
    for i, ch_id in enumerate(channel_ids[:max_channels]):
        print(f"Processing channel {ch_id} ({i+1}/{min(max_channels, len(channel_ids))})...")
        
        lfp_signal = lfp_matrix[i]
        
        # Detect ripples
        try:
            det_result = detect_ripples(
                lfp_signal, fs=fs,
                rp_band=(100, 140),
                order=550,
                window_ms=20,
                z_low=2.5,
                z_outlier=9.0,
                min_dur_ms=30,
                merge_dur_ms=10,
                epoch_ms=window_ms
            )
            
            if len(det_result.peak_idx) == 0:
                print(f"  No ripples detected for channel {ch_id}")
                continue
            
            # Create visualizations
            _create_channel_visualizations(
                lfp_signal, det_result, fs, ch_id, 
                output_path, "bipolar", window_ms
            )
            
        except Exception as e:
            print(f"  Error processing channel {ch_id}: {e}")
            continue
    
    print(f"✓ Bipolar visualizations saved to: {output_path}")


def visualize_car_ripples(car_dir: str, output_dir: str, 
                         max_channels_per_bank: int = 5, window_ms: int = 200):
    """
    Create comprehensive ripple visualizations for CAR re-referenced data.
    
    Args:
        car_dir: Directory containing CAR-processed data
        output_dir: Directory to save visualizations
        max_channels_per_bank: Maximum channels per bank to visualize
        window_ms: Window size for ripple extraction
    """
    print("=== CAR Ripple Visualization ===")
    
    car_path = Path(car_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all bank directories
    bank_dirs = sorted([d for d in car_path.iterdir() if d.is_dir() and d.name.startswith("bank_")])
    
    total_channels = 0
    for bank_dir in bank_dirs:
        bank_num = int(bank_dir.name.split("_")[1])
        print(f"\nProcessing Bank {bank_num}...")
        
        # Find channel directories in this bank
        ch_dirs = sorted([d for d in bank_dir.iterdir() if d.is_dir() and d.name.startswith("chan")])
        
        for i, ch_dir in enumerate(ch_dirs[:max_channels_per_bank]):
            try:
                # Extract channel number
                ch_name = ch_dir.name
                if "_car" in ch_name:
                    ch_id = int(ch_name.split("chan")[1].split("_car")[0])
                else:
                    ch_id = int(ch_name[4:])
                
                # Load CAR LFP data
                mat_file = ch_dir / "lfp_car.mat"
                if not mat_file.exists():
                    continue
                
                data = sio.loadmat(mat_file, squeeze_me=True)
                lfp_car = data["lfp_car"]
                fs = data.get("fs", 1000.0)
                
                if np.isnan(fs) or fs <= 0:
                    fs = 1000.0
                
                # Detect ripples
                det_result = detect_ripples(
                    lfp_car, fs=fs,
                    rp_band=(100, 140),
                    order=550,
                    window_ms=20,
                    z_low=2.5,
                    z_outlier=9.0,
                    min_dur_ms=30,
                    merge_dur_ms=10,
                    epoch_ms=window_ms
                )
                
                if len(det_result.peak_idx) == 0:
                    print(f"  No ripples detected for channel {ch_id}")
                    continue
                
                # Create visualizations
                _create_channel_visualizations(
                    lfp_car, det_result, fs, ch_id, 
                    output_path, "car", window_ms, bank_num
                )
                
                total_channels += 1
                
            except Exception as e:
                print(f"  Error processing {ch_dir}: {e}")
                continue
    
    print(f"\n✓ CAR visualizations saved to: {output_path}")
    print(f"Total channels processed: {total_channels}")


def _create_channel_visualizations(lfp: np.ndarray, det_result, fs: float, 
                                  ch_id: int, output_path: Path, 
                                  method: str, window_ms: int, bank_num: int = None):
    """
    Create all visualization figures for a single channel.
    
    Args:
        lfp: LFP signal
        det_result: Detection result object
        fs: Sampling frequency
        ch_id: Channel ID
        output_path: Output directory
        method: "bipolar" or "car"
        window_ms: Window size in milliseconds
        bank_num: Bank number (for CAR only)
    """
    # Extract ripple windows
    windowed_lfp, time_axis = _extract_ripple_windows(lfp, det_result.peak_idx, fs, window_ms)
    
    if windowed_lfp.size == 0:
        return
    
    # Compute spectrograms and spectra
    spectrograms, freq_axis, norm_spectrograms = _compute_spectrograms(windowed_lfp, fs)
    spectra, spec_freq = _compute_spectra(windowed_lfp, fs)
    
    # Create output directory for this channel
    if method == "bipolar":
        ch_output_dir = output_path / f"chan{ch_id:03d}_bipolar"
    else:
        ch_output_dir = output_path / f"bank{bank_num:02d}_chan{ch_id:03d}_car"
    
    ch_output_dir.mkdir(exist_ok=True)
    
    # Figure 1: Grand Average
    _create_grand_average_figure(
        windowed_lfp, time_axis, norm_spectrograms, freq_axis,
        ch_output_dir, ch_id, method, bank_num
    )
    
    # Figure 2: Individual Ripple Spectrograms
    _create_ripple_spectrograms_figure(
        windowed_lfp, time_axis, norm_spectrograms, freq_axis,
        ch_output_dir, ch_id, method, bank_num
    )
    
    # Figure 3: Individual Ripple Spectra
    _create_ripple_spectra_figure(
        spectra, spec_freq, ch_output_dir, ch_id, method, bank_num
    )


def _create_grand_average_figure(windowed_lfp: np.ndarray, time_axis: np.ndarray,
                                norm_spectrograms: np.ndarray, freq_axis: np.ndarray,
                                output_dir: Path, ch_id: int, method: str, bank_num: int = None):
    """Create grand average figure (accepted ripples)."""
    if windowed_lfp.size == 0 or norm_spectrograms.size == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 14))
    fig.suptitle(f'Grand Average Ripples - Channel {ch_id} ({method.upper()})', fontsize=14, fontweight='bold')
    
    # Grand average spectrogram
    avg_spectrogram = np.mean(norm_spectrograms, axis=0)
    im1 = ax1.imshow(avg_spectrogram, aspect='auto', origin='lower', 
                     extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]], 
                     cmap='hot')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Grand Average Spectrogram')
    ax1.set_xlabel('Time (ms)')
    
    # Grand average LFP
    avg_lfp = np.mean(windowed_lfp, axis=1)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_axis, avg_lfp, 'w', linewidth=1)
    ax2_twin.set_ylabel('Voltage (μV)')
    
    # Colorbar
    cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label('Normalized Power')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grand_average.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_ripple_spectrograms_figure(windowed_lfp: np.ndarray, time_axis: np.ndarray,
                                      norm_spectrograms: np.ndarray, freq_axis: np.ndarray,
                                      output_dir: Path, ch_id: int, method: str, bank_num: int = None):
    """Create individual ripple spectrograms figure."""
    if windowed_lfp.size == 0 or norm_spectrograms.size == 0:
        return
    
    n_ripples = windowed_lfp.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_ripples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Individual Ripple Spectrograms - Channel {ch_id} ({method.upper()})', 
                 fontsize=14, fontweight='bold')
    
    for i in range(n_ripples):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Spectrogram
        im = ax.imshow(norm_spectrograms[i], aspect='auto', origin='lower',
                       extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]], 
                       cmap='hot')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Ripple {i+1}')
        
        # LFP trace
        ax_twin = ax.twinx()
        ax_twin.plot(time_axis, windowed_lfp[:, i], 'w', linewidth=1)
        ax_twin.set_ylabel('Voltage (μV)')
    
    # Hide empty subplots
    for i in range(n_ripples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes.ravel(), fraction=0.046, pad=0.02)
    cbar.set_label('Normalized Power')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accepted_spectrogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_ripple_spectra_figure(spectra: np.ndarray, freq_axis: np.ndarray,
                                 output_dir: Path, ch_id: int, method: str, bank_num: int = None):
    """Create individual ripple spectra figure."""
    if spectra.size == 0:
        return
    
    n_ripples = spectra.shape[0]
    n_cols = 4
    n_rows = int(np.ceil(n_ripples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Individual Ripple Spectra - Channel {ch_id} ({method.upper()})', 
                 fontsize=14, fontweight='bold')
    
    for i in range(n_ripples):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot spectrum
        ax.plot(freq_axis, spectra[i], linewidth=1.2)
        ax.axvline(100, color='r', linestyle='--', linewidth=1)
        ax.axvline(140, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (μV²)')
        ax.set_title(f'Ripple {i+1}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_ripples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accepted_visual_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_visualization(bipolar_dir: str, car_dir: str, output_dir: str):
    """
    Create summary visualization comparing bipolar and CAR methods.
    
    Args:
        bipolar_dir: Directory containing bipolar results
        car_dir: Directory containing CAR results
        output_dir: Directory to save summary
    """
    print("=== Creating Summary Visualization ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    bipolar_summary = sio.loadmat(Path(bipolar_dir) / "bipolar_lfp_matrix_ripples" / "bipolar_ripple_summary.mat", squeeze_me=True)
    car_summary = sio.loadmat(Path(car_dir) / "car_ripple_summary.mat", squeeze_me=True)
    
    # Create comparison figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bipolar vs CAR Re-referencing Comparison', fontsize=16, fontweight='bold')
    
    # Ripple counts comparison
    bipolar_counts = [bipolar_summary['channel_summary'][f'channel_{ch}']['n_ripples'] 
                     for ch in bipolar_summary['channel_summary'].keys()]
    car_counts = [car_summary['channel_summary'][f'channel_{ch}']['n_ripples'] 
                 for ch in car_summary['channel_summary'].keys()]
    
    ax1.hist(bipolar_counts, bins=20, alpha=0.7, label='Bipolar', color='blue')
    ax1.hist(car_counts, bins=20, alpha=0.7, label='CAR', color='red')
    ax1.set_xlabel('Ripples per Channel')
    ax1.set_ylabel('Number of Channels')
    ax1.set_title('Ripple Count Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Total ripples comparison
    methods = ['Bipolar', 'CAR']
    totals = [bipolar_summary['total_ripples'], car_summary['total_ripples']]
    bars = ax2.bar(methods, totals, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Total Ripples')
    ax2.set_title('Total Ripples Detected')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, total in zip(bars, totals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.01,
                f'{total:,}', ha='center', va='bottom', fontweight='bold')
    
    # Channels processed
    channels = [bipolar_summary['n_channels_processed'], car_summary['n_channels_processed']]
    bars = ax3.bar(methods, channels, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Channels Processed')
    ax3.set_title('Channels Processed')
    ax3.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, channels):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(channels)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Average ripples per channel
    avg_ripples = [np.mean(bipolar_counts), np.mean(car_counts)]
    bars = ax4.bar(methods, avg_ripples, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Average Ripples per Channel')
    ax4.set_title('Average Ripples per Channel')
    ax4.grid(True, alpha=0.3)
    
    for bar, avg in zip(bars, avg_ripples):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_ripples)*0.01,
                f'{avg:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary visualization saved to: {output_path}")


__all__ = [
    "visualize_bipolar_ripples",
    "visualize_car_ripples", 
    "create_summary_visualization"
]
