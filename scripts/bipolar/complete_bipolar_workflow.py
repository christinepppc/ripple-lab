#!/usr/bin/env python3
"""
Complete Bipolar Processing and Ripple Analysis Workflow

This script provides a unified workflow for:
1. Bipolar re-referencing (single trial or batch)
2. Ripple detection, normalization, and rejection
3. Comprehensive statistics and reporting

Usage:
    # Single trial processing
    python scripts/bipolar/complete_bipolar_workflow.py single /path/to/trial/dir --output /path/to/output
    
    # Batch processing
    python scripts/bipolar/complete_bipolar_workflow.py batch --base-dir /path/to/data --sessions 134,135,136
    
    # Ripple analysis only (on already processed data)
    python scripts/bipolar/complete_bipolar_workflow.py analyze --matrix /path/to/bipolar_lfp_matrix.mat
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import json
import time
from datetime import datetime

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from ripple_core.ripple_core.signal import process_bipolar_referencing, get_bipolar_lfp_matrix
from ripple_core.ripple_core.signal.pairs import WHITE_MATTER_CHANNELS
from ripple_core.ripple_core.analyze import detect_ripples, normalize_ripples, reject_ripples


def process_single_trial(trial_dir: Path, output_dir: Path, bad_channels: list = None) -> dict:
    """Process a single trial for bipolar referencing."""
    print(f"Processing single trial: {trial_dir}")
    
    if bad_channels is None:
        bad_channels = []
    
    # Process bipolar referencing
    result = process_bipolar_referencing(
        trial_dir=trial_dir,
        output_dir=output_dir,
        bad_channels=bad_channels,
        white_matter_channels=WHITE_MATTER_CHANNELS
    )
    
    return result


def process_batch_trials(base_dir: Path, sessions: list, trials: list, output_dir: Path, bad_channels: list = None) -> dict:
    """Process multiple trials in batch."""
    print(f"Processing batch: {len(sessions)} sessions, {len(trials)} trials each")
    
    if bad_channels is None:
        bad_channels = []
    
    all_results = []
    total_processed = 0
    total_failed = 0
    
    for session in sessions:
        for trial in trials:
            trial_dir = base_dir / f"session{session}" / f"trial{trial:03d}"
            
            if not trial_dir.exists():
                print(f"  Skipping {trial_dir} (not found)")
                continue
            
            print(f"  Processing session{session}/trial{trial:03d}...")
            
            try:
                result = process_bipolar_referencing(
                    trial_dir=trial_dir,
                    output_dir=output_dir,
                    bad_channels=bad_channels,
                    white_matter_channels=WHITE_MATTER_CHANNELS
                )
                all_results.append(result)
                total_processed += 1
                print(f"    ✓ Success")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                total_failed += 1
                all_results.append({
                    'trial_dir': str(trial_dir),
                    'error': str(e),
                    'status': 'failed'
                })
    
    return {
        'total_processed': total_processed,
        'total_failed': total_failed,
        'results': all_results
    }


def analyze_ripples(matrix_file: Path, output_dir: Path, ripple_params: dict = None) -> dict:
    """Analyze ripples in bipolar LFP matrix."""
    print(f"Analyzing ripples in: {matrix_file}")
    
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
    
    # Load the LFP matrix
    data = sio.loadmat(matrix_file)
    lfp_matrix = data['lfp_matrix']
    channel_ids = data['channel_ids'].flatten()
    fs = float(data.get('fs', 1000))
    
    print(f"  Loaded {lfp_matrix.shape[0]} channels, {lfp_matrix.shape[1]} samples")
    print(f"  Sampling rate: {fs} Hz, Duration: {lfp_matrix.shape[1] / fs:.1f}s")
    
    # Process each channel
    channel_results = []
    total_ripples = 0
    total_passed = 0
    total_rejected = 0
    
    print("  Processing channels...")
    
    for i, ch_id in enumerate(channel_ids):
        if i % 20 == 0:  # Progress indicator
            print(f"    Channel {i+1}/{len(channel_ids)}...")
        
        lfp_data = lfp_matrix[i, :]
        
        try:
            # Detect ripples
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
                channel_results.append({
                    'channel_id': int(ch_id),
                    'n_ripples_detected': 0,
                    'n_ripples_passed': 0,
                    'n_ripples_rejected': 0,
                    'pass_rate': 0.0
                })
                continue
            
            # Normalize ripples
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
            
            # Reject ripples
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
            
            channel_results.append({
                'channel_id': int(ch_id),
                'n_ripples_detected': n_ripples,
                'n_ripples_passed': n_passed,
                'n_ripples_rejected': n_rejected,
                'pass_rate': pass_rate
            })
            
            total_ripples += n_ripples
            total_passed += n_passed
            total_rejected += n_rejected
            
        except Exception as e:
            print(f"    Error in channel {ch_id}: {e}")
            channel_results.append({
                'channel_id': int(ch_id),
                'n_ripples_detected': 0,
                'n_ripples_passed': 0,
                'n_ripples_rejected': 0,
                'pass_rate': 0.0,
                'error': str(e)
            })
    
    # Calculate summary statistics
    successful_channels = [r for r in channel_results if 'error' not in r]
    channels_with_ripples = [r for r in channel_results if r['n_ripples_detected'] > 0]
    
    if successful_channels:
        ripple_counts = [r['n_ripples_detected'] for r in successful_channels]
        pass_rates = [r['pass_rate'] for r in successful_channels]
        
        overall_pass_rate = total_passed / total_ripples if total_ripples > 0 else 0
        
        summary = {
            'matrix_file': str(matrix_file),
            'n_channels': len(channel_ids),
            'n_channels_processed': len(successful_channels),
            'n_channels_with_ripples': len(channels_with_ripples),
            'n_channels_without_ripples': len(channel_results) - len(channels_with_ripples),
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
            'min_ripples_in_channel': np.min(ripple_counts),
            'ripple_params': ripple_params,
            'processing_time': datetime.now().isoformat()
        }
    else:
        summary = {
            'matrix_file': str(matrix_file),
            'error': 'No channels processed successfully',
            'processing_time': datetime.now().isoformat()
        }
    
    summary['channel_results'] = channel_results
    
    # Save results
    output_file = output_dir / f"ripple_analysis_{matrix_file.stem}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\nRipple Analysis Summary:")
    print(f"  Channels processed: {summary.get('n_channels_processed', 0)}")
    print(f"  Channels with ripples: {summary.get('n_channels_with_ripples', 0)}")
    print(f"  Total ripples detected: {summary.get('total_ripples_detected', 0):,}")
    print(f"  Total ripples passed: {summary.get('total_ripples_passed', 0):,}")
    print(f"  Overall pass rate: {summary.get('overall_pass_rate', 0):.1%}")
    print(f"  Results saved to: {output_file}")
    
    return summary


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Complete Bipolar Processing and Ripple Analysis Workflow")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single trial processing
    single_parser = subparsers.add_parser('single', help='Process a single trial')
    single_parser.add_argument('trial_dir', type=Path, help='Path to trial directory')
    single_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    single_parser.add_argument('--bad-channels', type=str, help='Comma-separated list of bad channels')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple trials in batch')
    batch_parser.add_argument('--base-dir', type=Path, required=True, help='Base data directory')
    batch_parser.add_argument('--sessions', type=str, required=True, help='Comma-separated session numbers')
    batch_parser.add_argument('--trials', type=str, required=True, help='Comma-separated trial numbers')
    batch_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    batch_parser.add_argument('--bad-channels', type=str, help='Comma-separated list of bad channels')
    
    # Ripple analysis
    analyze_parser = subparsers.add_parser('analyze', help='Analyze ripples in bipolar matrix')
    analyze_parser.add_argument('--matrix', type=Path, required=True, help='Path to bipolar LFP matrix file')
    analyze_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        bad_channels = []
        if args.bad_channels:
            bad_channels = [int(x) for x in args.bad_channels.split(',')]
        
        result = process_single_trial(args.trial_dir, args.output, bad_channels)
        print(f"Single trial processing complete. Results: {result}")
        
    elif args.command == 'batch':
        bad_channels = []
        if args.bad_channels:
            bad_channels = [int(x) for x in args.bad_channels.split(',')]
        
        sessions = [int(x) for x in args.sessions.split(',')]
        trials = [int(x) for x in args.trials.split(',')]
        
        result = process_batch_trials(args.base_dir, sessions, trials, args.output, bad_channels)
        print(f"Batch processing complete. Processed: {result['total_processed']}, Failed: {result['total_failed']}")
        
    elif args.command == 'analyze':
        args.output.mkdir(parents=True, exist_ok=True)
        result = analyze_ripples(args.matrix, args.output)
        print(f"Ripple analysis complete.")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
