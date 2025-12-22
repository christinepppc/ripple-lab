#!/usr/bin/env python3
"""
Simple ripple analysis on bipolar LFP matrix.

This script loads the bipolar LFP matrix and runs ripple detection,
normalization, and rejection on each channel.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import json
from datetime import datetime
import time

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from ripple_core.ripple_core.analyze import detect_ripples, normalize_ripples, reject_ripples


def analyze_bipolar_matrix(matrix_file: Path, output_dir: Path, ripple_params: dict = None) -> dict:
    """Analyze bipolar LFP matrix for ripples."""
    
    if ripple_params is None:
        ripple_params = {
            'fs': 1000,
            'rp_band': (100, 140),
            'order': 550,
            'window_ms': 20,
            'z_low': 2.5,
            'z_outlier': 9.0,
            'min_dur_ms': 30,
            'merge_dur_ms': 10,
            'epoch_ms': 200,
            'fmin': 2,
            'fmax': 200,
            'win_length': 0.060,
            'step': 0.001,
            'nw': 1.2,
            'tapers': 1,
            'tfspec_pad': 20,
            'reject_thresh': 3.0
        }
    
    print(f"Loading bipolar LFP matrix from: {matrix_file}")
    
    # Load the LFP matrix
    data = sio.loadmat(matrix_file)
    lfp_matrix = data['lfp_matrix']  # Shape: (n_channels, n_samples)
    channel_ids = data['channel_ids'].flatten()  # Channel IDs
    fs = float(data.get('fs', 1000))  # Sampling frequency
    
    print(f"Loaded {lfp_matrix.shape[0]} bipolar channels, {lfp_matrix.shape[1]} samples")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {lfp_matrix.shape[1] / fs:.1f} seconds")
    print()
    
    # Initialize results
    results = {
        'matrix_file': str(matrix_file),
        'n_channels': lfp_matrix.shape[0],
        'n_samples': lfp_matrix.shape[1],
        'fs': fs,
        'duration_sec': lfp_matrix.shape[1] / fs,
        'channel_ids': channel_ids.tolist(),
        'ripple_params': ripple_params,
        'processing_time': datetime.now().isoformat()
    }
    
    # Process each channel
    all_channel_results = []
    total_ripples = 0
    total_passed = 0
    total_rejected = 0
    
    print("Processing channels...")
    print("=" * 50)
    
    for i, ch_id in enumerate(channel_ids):
        print(f"Channel {i+1:3d}/{len(channel_ids)} (ID: {ch_id:3d}): ", end="")
        
        # Get LFP data for this channel
        lfp_data = lfp_matrix[i, :]
        
        try:
            # Step 1: Detect ripples
            det_res = detect_ripples(
                lfp_data,
                fs=ripple_params['fs'],
                rp_band=ripple_params['rp_band'],
                order=ripple_params['order'],
                window_ms=ripple_params['window_ms'],
                z_low=ripple_params['z_low'],
                z_outlier=ripple_params['z_outlier'],
                min_dur_ms=ripple_params['min_dur_ms'],
                merge_dur_ms=ripple_params['merge_dur_ms'],
                epoch_ms=ripple_params['epoch_ms']
            )
            
            n_ripples = len(det_res.peak_idx)
            
            if n_ripples == 0:
                print("No ripples")
                all_channel_results.append({
                    'channel_id': int(ch_id),
                    'n_ripples_detected': 0,
                    'n_ripples_passed': 0,
                    'n_ripples_rejected': 0,
                    'pass_rate': 0.0,
                    'status': 'no_ripples'
                })
                continue
            
            # Step 2: Normalize ripples
            norm_res = normalize_ripples(
                lfp_data,
                fs=ripple_params['fs'],
                raw_windowed_lfp=det_res.raw_windowed_lfp,
                real_duration=det_res.real_duration,
                fmin=ripple_params['fmin'],
                fmax=ripple_params['fmax'],
                win_length=ripple_params['win_length'],
                step=ripple_params['step'],
                nw=ripple_params['nw'],
                tapers=ripple_params['tapers'],
                tfspec_pad=ripple_params['tfspec_pad']
            )
            
            # Step 3: Reject ripples
            rej_res = reject_ripples(
                freq_spec_actual=norm_res.freq_spec_actual,
                spec_f=norm_res.spec_f,
                mu=det_res.mu,
                sd=det_res.sd,
                strict_threshold=ripple_params['reject_thresh'],
                env_rip=det_res.env_rip,
                peak_idx=det_res.peak_idx
            )
            
            n_passed = len(rej_res.pass_idx)
            n_rejected = len(rej_res.reject_idx)
            pass_rate = n_passed / n_ripples if n_ripples > 0 else 0
            
            print(f"{n_ripples:3d} ripples -> {n_passed:3d} passed ({pass_rate:.1%})")
            
            # Store results
            channel_result = {
                'channel_id': int(ch_id),
                'n_ripples_detected': n_ripples,
                'n_ripples_passed': n_passed,
                'n_ripples_rejected': n_rejected,
                'pass_rate': pass_rate,
                'status': 'success'
            }
            
            all_channel_results.append(channel_result)
            total_ripples += n_ripples
            total_passed += n_passed
            total_rejected += n_rejected
            
        except Exception as e:
            print(f"Error: {e}")
            all_channel_results.append({
                'channel_id': int(ch_id),
                'n_ripples_detected': 0,
                'n_ripples_passed': 0,
                'n_ripples_rejected': 0,
                'pass_rate': 0.0,
                'status': 'error',
                'error': str(e)
            })
    
    # Calculate summary statistics
    successful_channels = [r for r in all_channel_results if r['status'] == 'success']
    channels_with_ripples = [r for r in all_channel_results if r['n_ripples_detected'] > 0]
    channels_without_ripples = [r for r in all_channel_results if r['n_ripples_detected'] == 0]
    
    if successful_channels:
        ripple_counts = [r['n_ripples_detected'] for r in successful_channels]
        pass_rates = [r['pass_rate'] for r in successful_channels]
        
        overall_pass_rate = total_passed / total_ripples if total_ripples > 0 else 0
        
        results.update({
            'n_channels_processed': len(successful_channels),
            'n_channels_with_ripples': len(channels_with_ripples),
            'n_channels_without_ripples': len(channels_without_ripples),
            'total_ripples_detected': total_ripples,
            'total_ripples_passed': total_passed,
            'total_ripples_rejected': total_rejected,
            'overall_pass_rate': overall_pass_rate,
            'mean_ripples_per_channel': np.mean(ripple_counts),
            'std_ripples_per_channel': np.std(ripple_counts),
            'median_ripples_per_channel': np.median(ripple_counts),
            'mean_pass_rate_per_channel': np.mean(pass_rates),
            'std_pass_rate_per_channel': np.std(pass_rates),
            'median_pass_rate_per_channel': np.median(pass_rates),
            'max_ripples_in_channel': np.max(ripple_counts),
            'min_ripples_in_channel': np.min(ripple_counts)
        })
    else:
        results.update({
            'n_channels_processed': 0,
            'total_ripples_detected': 0,
            'total_ripples_passed': 0,
            'total_ripples_rejected': 0,
            'overall_pass_rate': 0,
            'error': 'No channels processed successfully'
        })
    
    results['channel_results'] = all_channel_results
    
    # Save results
    output_file = output_dir / "ripple_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("Ripple Analysis Summary")
    print("=" * 50)
    print(f"Channels processed: {results.get('n_channels_processed', 0)}")
    print(f"Channels with ripples: {results.get('n_channels_with_ripples', 0)}")
    print(f"Channels without ripples: {results.get('n_channels_without_ripples', 0)}")
    print(f"Total ripples detected: {results.get('total_ripples_detected', 0):,}")
    print(f"Total ripples passed: {results.get('total_ripples_passed', 0):,}")
    print(f"Total ripples rejected: {results.get('total_ripples_rejected', 0):,}")
    print(f"Overall pass rate: {results.get('overall_pass_rate', 0):.1%}")
    
    if successful_channels:
        print(f"Mean ripples per channel: {results.get('mean_ripples_per_channel', 0):.1f} ± {results.get('std_ripples_per_channel', 0):.1f}")
        print(f"Median ripples per channel: {results.get('median_ripples_per_channel', 0):.1f}")
        print(f"Mean pass rate per channel: {results.get('mean_pass_rate_per_channel', 0):.1%} ± {results.get('std_pass_rate_per_channel', 0):.1%}")
        print(f"Median pass rate per channel: {results.get('median_pass_rate_per_channel', 0):.1%}")
        print(f"Max ripples in single channel: {results.get('max_ripples_in_channel', 0)}")
        print(f"Min ripples in single channel: {results.get('min_ripples_in_channel', 0)}")
    
    print(f"Results saved to: {output_file}")
    
    return results


def main():
    """Main function."""
    # Set up paths
    matrix_file = Path("/vol/cortex/cd4/pesaranlab/ripple-lab/results/bipolar_processed/session134_trial001_bipolar_processed/bipolar_lfp_matrix.mat")
    output_dir = Path("/vol/cortex/cd4/pesaranlab/ripple-lab/results/ripple_analysis")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ripple detection parameters
    ripple_params = {
        'fs': 1000,
        'rp_band': (100, 140),
        'order': 550,
        'window_ms': 20,
        'z_low': 2.5,
        'z_outlier': 9.0,
        'min_dur_ms': 30,
        'merge_dur_ms': 10,
        'epoch_ms': 200,
        'fmin': 2,
        'fmax': 200,
        'win_length': 0.060,
        'step': 0.001,
        'nw': 1.2,
        'tapers': 1,
        'tfspec_pad': 20,
        'reject_thresh': 3.0
    }
    
    # Run the analysis
    start_time = time.time()
    results = analyze_bipolar_matrix(matrix_file, output_dir, ripple_params)
    processing_time = time.time() - start_time
    
    print(f"\nTotal processing time: {processing_time:.1f} seconds")
    
    return results


if __name__ == "__main__":
    main()
