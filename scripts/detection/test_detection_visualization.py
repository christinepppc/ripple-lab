#!/usr/bin/env python3
"""
Test ripple detection and visualization on a single channel.

Usage:
    python scripts/test_detection_visualization.py --session_idx 46 --trial 1 --channel 1
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add ripple_core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'ripple_core'))

from ripple_core.load import load_movie_database, load_electrodes
from ripple_core.analyze import detect_ripples, normalize_ripples
from ripple_core.visualize import plot_processing_notebook


def main():
    parser = argparse.ArgumentParser(description="Test ripple detection and visualization")
    parser.add_argument('--session_idx', type=int, default=46, help='Session index (0-based)')
    parser.add_argument('--trial', type=int, default=1, help='Trial number')
    parser.add_argument('--channel', type=int, default=1, help='Channel number')
    parser.add_argument('--output', type=str, default='outputs/test_detection', help='Output directory')
    parser.add_argument('--z_low', type=float, default=3.0, help='Detection threshold (z-score)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get session info
    sessions = load_movie_database()
    sess = sessions[args.session_idx]
    sess_number = sess['session']
    sess_date = sess['date']
    
    print("="*80)
    print(f"RIPPLE DETECTION TEST")
    print("="*80)
    print(f"Session: {sess_number} (index={args.session_idx}, date={sess_date})")
    print(f"Trial: {args.trial}")
    print(f"Channel: {args.channel}")
    print(f"Detection threshold (z_low): {args.z_low}")
    print("="*80)
    
    # Load LFP data
    print("\n[1/3] Loading LFP data...")
    lfp_data = load_electrodes(args.session_idx, args.trial, args.channel)
    n_samples = len(lfp_data)
    duration_sec = n_samples / 1000.0
    print(f"  ✓ Loaded {n_samples:,} samples ({duration_sec:.1f} seconds)")
    
    # Detect ripples
    print("\n[2/3] Detecting ripples...")
    det_res = detect_ripples(
        lfp_data,
        fs=1000,
        rp_band=(100, 140),
        order=550,
        window_ms=20.0,
        z_low=args.z_low,
        z_outlier=9.0,
        min_dur_ms=30.0,
        merge_dur_ms=10.0,
        epoch_ms=200
    )
    
    n_ripples = len(det_res.peak_idx)
    print(f"  ✓ Detected {n_ripples} ripples")
    
    if n_ripples > 0:
        peak_times_sec = det_res.peak_idx / 1000.0
        durations_ms = (det_res.real_duration[:, 1] - det_res.real_duration[:, 0]) / 1000.0 * 1000
        print(f"  - Mean duration: {durations_ms.mean():.1f} ms")
        print(f"  - First ripple at: {peak_times_sec[0]:.2f} s")
        print(f"  - Last ripple at: {peak_times_sec[-1]:.2f} s")
    
    # Compute time-frequency decomposition (spectrograms)
    norm_res = None
    if n_ripples > 0:
        print("\n[2.5/3] Computing time-frequency decomposition...")
        try:
            norm_res = normalize_ripples(
                lfp_data,
                fs=1000,
                raw_windowed_lfp=det_res.raw_windowed_lfp,
                real_duration=det_res.real_duration,
                fmin=2,
                fmax=200,
                win_length=0.060,
                step=0.001,
                nw=1.2,
                tapers=2,
                tfspec_pad=20
            )
            print(f"  ✓ Computed spectrograms for {n_ripples} ripples")
        except Exception as e:
            print(f"  ✗ Failed to compute spectrograms: {e}")
            norm_res = None
    
    # Create visualizations
    print("\n[3/3] Creating visualizations...")
    
    # Main processing notebook view
    fig = plt.figure(figsize=(16, 10))
    plot_processing_notebook(
        fig,
        lfp_data,
        fs=1000,
        bp_lfp=det_res.bp_lfp,
        env_rip=det_res.env_rip,
        event_bounds=det_res.real_duration,
        fmax=200
    )
    
    output_file = output_dir / f"detection_sess{sess_number}_trial{args.trial:03d}_ch{args.channel:03d}_zlow{args.z_low:.1f}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close(fig)
    
    # Detailed view of first few ripples
    if n_ripples > 0:
        n_show = min(6, n_ripples)
        n_cols = 4 if norm_res is not None else 3
        fig, axes = plt.subplots(n_show, n_cols, figsize=(5*n_cols, 3*n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f"First {n_show} Ripples - Session {sess_number}, Trial {args.trial}, Channel {args.channel}", 
                     fontsize=14, fontweight='bold')
        
        for i in range(n_show):
            # Get ripple window
            peak_idx = det_res.peak_idx[i]
            start_idx = max(0, peak_idx - 100)
            end_idx = min(len(lfp_data), peak_idx + 100)
            time_ms = (np.arange(start_idx, end_idx) - peak_idx) * 1000 / 1000.0
            
            # Raw LFP
            axes[i, 0].plot(time_ms, lfp_data[start_idx:end_idx], 'k-', linewidth=0.5)
            axes[i, 0].axvline(0, color='r', linestyle='--', alpha=0.5)
            axes[i, 0].set_ylabel(f'Ripple {i+1}\nRaw LFP (μV)')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].set_title('Raw LFP')
            if i == n_show - 1:
                axes[i, 0].set_xlabel('Time (ms)')
            
            # Bandpass filtered
            axes[i, 1].plot(time_ms, det_res.bp_lfp[start_idx:end_idx], 'b-', linewidth=0.8)
            axes[i, 1].axvline(0, color='r', linestyle='--', alpha=0.5)
            axes[i, 1].set_ylabel('BP 100-140 Hz (μV)')
            axes[i, 1].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 1].set_title('Bandpass Filtered')
            if i == n_show - 1:
                axes[i, 1].set_xlabel('Time (ms)')
            
            # RMS envelope
            axes[i, 2].plot(time_ms, det_res.env_rip[start_idx:end_idx], 'g-', linewidth=1.0)
            axes[i, 2].axhline(det_res.mu + args.z_low * det_res.sd, color='r', 
                              linestyle='--', alpha=0.5, label='Threshold')
            axes[i, 2].axvline(0, color='r', linestyle='--', alpha=0.5)
            axes[i, 2].set_ylabel('RMS Envelope (μV)')
            axes[i, 2].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 2].set_title('RMS Envelope')
                axes[i, 2].legend(loc='upper right', fontsize=8)
            if i == n_show - 1:
                axes[i, 2].set_xlabel('Time (ms)')
            
            # Spectrogram (if available)
            if norm_res is not None:
                # Get spectrogram for this ripple (F x T x N format)
                spec = norm_res.normalized_ripple_windowed[:, :, i]  # F x T
                
                # Time axis for spectrogram (centered on ripple peak)
                epoch_ms = 200  # from det_res.epoch_ms
                t_spec = np.linspace(-epoch_ms, epoch_ms, spec.shape[1])
                
                # Frequency axis
                f_spec = norm_res.spec_f
                
                # Plot spectrogram
                im = axes[i, 3].imshow(
                    spec, 
                    aspect='auto', 
                    origin='lower',
                    extent=[t_spec[0], t_spec[-1], f_spec[0], f_spec[-1]],
                    cmap='jet',
                    interpolation='bilinear'
                )
                axes[i, 3].axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=1)
                axes[i, 3].set_ylabel('Frequency (Hz)')
                axes[i, 3].set_ylim([50, 200])  # Focus on ripple band
                if i == 0:
                    axes[i, 3].set_title('Spectrogram')
                if i == n_show - 1:
                    axes[i, 3].set_xlabel('Time (ms)')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i, 3], label='Normalized Power')
        
        plt.tight_layout()
        detail_file = output_dir / f"ripple_details_sess{sess_number}_trial{args.trial:03d}_ch{args.channel:03d}_zlow{args.z_low:.1f}.png"
        fig.savefig(detail_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {detail_file}")
        plt.close(fig)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total ripples detected: {n_ripples}")
    if n_ripples > 0:
        print(f"Ripple rate: {n_ripples / duration_sec * 60:.2f} ripples/min")
        print(f"Mean duration: {durations_ms.mean():.1f} ± {durations_ms.std():.1f} ms")
        print(f"Detection threshold: μ + {args.z_low}σ = {det_res.mu:.2f} + {args.z_low}×{det_res.sd:.2f} = {det_res.mu + args.z_low * det_res.sd:.2f} μV")
    print(f"\nVisualizations saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
