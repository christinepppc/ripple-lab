#!/usr/bin/env python3
"""
Count detected vs rejected ripples for specified sessions
"""

import pandas as pd
import scipy.io as sio
from pathlib import Path
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Count rejected ripples')
parser.add_argument('--sessions', type=int, nargs='+', required=True, help='Session numbers')
args = parser.parse_args()

BASE_DIR = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")

# Session-trial mapping (from the pipeline)
SESSION_TRIALS = {
    32: (3, 8),
    33: (5, 10),
    34: (5, 11),
    35: (6, 11),
    37: (2, 7),
    41: (5, 8),
    45: (5, 8),
    46: (7, 9),
    50: (2, 4),
    51: (5, 7),
    52: (2, 4),
    59: (6, 9),
    60: (6, 8),
    61: (10, 12),
    64: (10, 12),
    65: (5, 7),
    66: (5, 7),
    67: (5, 7),
    73: (2, 4),
    74: (2, 4),
    75: (2, 4),
    76: (2, 4),
    77: (1, 5),
    78: (2, 4),
    79: (2, 4),
    81: (2, 8),
    82: (2, 7),
    85: (2, 5),
    86: (2, 4),
}

def count_ripples_in_trial(session, trial):
    """Count detected and passed ripples for a trial"""
    trial_dir = BASE_DIR / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    
    # Try loading detection summary first (preferred method)
    summary_file = trial_dir / "ripple_detection_summary_zlow3.0.mat"
    
    n_detected = None
    n_channels_processed = None
    n_channels_with_ripples = None
    
    if summary_file.exists():
        mat_data = sio.loadmat(str(summary_file))
        if 'total_ripples' in mat_data:
            n_detected = int(mat_data['total_ripples'][0, 0])
            n_channels_processed = int(mat_data['n_channels_processed'][0, 0])
            n_channels_with_ripples = int(mat_data['channels_with_ripples'][0, 0])
    
    # If summary doesn't exist, count from individual channel files
    if n_detected is None:
        n_detected = 0
        n_channels_processed = 0
        n_channels_with_ripples = 0
        
        for ch_dir in sorted(trial_dir.glob("b*/")):
            if not ch_dir.is_dir():
                continue
            
            ch_file = ch_dir / f"ripples_{ch_dir.name}_zlow3.0.mat"
            if ch_file.exists():
                n_channels_processed += 1
                try:
                    ch_data = sio.loadmat(str(ch_file))
                    # Get number of ripples detected in this channel
                    ripple_count = 0
                    if 'n_ripples' in ch_data:
                        ripple_count = int(ch_data['n_ripples'][0, 0])
                    elif 'real_duration' in ch_data and ch_data['real_duration'].size > 0:
                        ripple_count = ch_data['real_duration'].shape[0]
                    
                    n_detected += ripple_count
                    if ripple_count > 0:
                        n_channels_with_ripples += 1
                except Exception as e:
                    pass
        
        # If we couldn't count any ripples, return None
        if n_channels_processed == 0:
            return None, None, None, None
    
    # Load passed ripples
    passed_file = trial_dir / "derived" / "ripples_zlow3.0_passed.pkl"
    if not passed_file.exists():
        return n_detected, None, n_channels_processed, n_channels_with_ripples
    
    passed_df = pd.read_pickle(passed_file)
    n_passed = len(passed_df)
    
    return n_detected, n_passed, n_channels_processed, n_channels_with_ripples

print("="*80)
print("RIPPLE DETECTION AND REJECTION STATISTICS")
print("="*80)

for session in sorted(args.sessions):
    if session not in SESSION_TRIALS:
        print(f"\nSession {session:03d}: Not in pipeline list")
        continue
    
    pre_trial, post_trial = SESSION_TRIALS[session]
    
    print(f"\n{'='*80}")
    print(f"SESSION {session:03d}")
    print(f"{'='*80}")
    
    for trial_type, trial in [("Pre", pre_trial), ("Post", post_trial)]:
        n_detected, n_passed, n_channels, n_active = count_ripples_in_trial(session, trial)
        
        if n_detected is None:
            print(f"\n{trial_type}-Stimulation (Trial {trial:03d}): No detection data found")
            continue
        
        if n_passed is None:
            print(f"\n{trial_type}-Stimulation (Trial {trial:03d}): No passed ripples data found")
            continue
        
        n_rejected = n_detected - n_passed
        rejection_rate = 100 * n_rejected / n_detected if n_detected > 0 else 0
        
        print(f"\n{trial_type}-Stimulation (Trial {trial:03d}):")
        print(f"  Channels processed:   {n_channels:3d}")
        print(f"  Channels with ripples: {n_active:3d}")
        print(f"  Detected (total):      {n_detected:5d}")
        print(f"  Passed (after reject): {n_passed:5d}")
        print(f"  Rejected:              {n_rejected:5d} ({rejection_rate:.1f}%)")

print("\n" + "="*80)
