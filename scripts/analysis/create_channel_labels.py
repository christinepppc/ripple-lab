#!/usr/bin/env python3
"""
Create channel label mappings for bipolar-referenced data.

This script reads the bipolar pairs and creates CSV files mapping:
1. Original channels (1-220) → anatomical labels
2. Bipolar channels (b001, b002, ...) → regions (prefrontal, parietal, motor, etc.)

Usage:
    python scripts/analysis/create_channel_labels.py --trial_bipolar_dir /path/to/trial_bipolar
"""

import sys
from pathlib import Path

# Add ripple_core to path (temporary - will be removed after proper install)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'packages' / 'ripple_core'))

import argparse
import scipy.io as sio
import pandas as pd
from ripple_core.labels import CHANNEL_LABELS, create_bipolar_channel_labels


def main():
    parser = argparse.ArgumentParser(
        description="Create channel label mappings for bipolar-referenced data"
    )
    parser.add_argument(
        '--session',
        type=int,
        required=True,
        help='Session number (e.g., 1, 46, 134)'
    )
    parser.add_argument(
        '--trial',
        type=int,
        required=True,
        help='Trial number (e.g., 1, 4, 6)'
    )
    args = parser.parse_args()
    
    # Construct trial bipolar directory
    base_dir = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    trial_bipolar_dir = base_dir / f"session{args.session:03d}" / f"trial{args.trial:03d}_bipolar"
    
    if not trial_bipolar_dir.exists():
        print(f"Error: Directory not found: {trial_bipolar_dir}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 1: Create channel labels CSV (original channels → anatomical labels)
    # ========================================================================
    
    channel_data = []
    for ch_id, label in CHANNEL_LABELS.items():
        channel_data.append({
            'channel': ch_id,
            'label': label if label else "unknown"
        })
    
    channel_df = pd.DataFrame(channel_data)
    
    # Save to the parent directory (session level)
    session_dir = trial_bipolar_dir.parent
    channel_labels_file = session_dir / "channel_labels.csv"
    channel_df.to_csv(channel_labels_file, index=False)
    
    print(f"Created channel labels CSV: {channel_labels_file}")
    print(f"Total channels: {len(channel_df)}")
    print(f"Channels with labels: {(channel_df['label'] != 'unknown').sum()}")
    
    # ========================================================================
    # STEP 2: Create bipolar channel labels CSV
    # ========================================================================
    
    # Load pairs_used.mat from the trial_bipolar directory
    pairs_file = trial_bipolar_dir / "pairs_used.mat"
    
    if not pairs_file.exists():
        print(f"Error: pairs_used.mat not found: {pairs_file}")
        sys.exit(1)
    
    # Load pairs data
    pairs_mat = sio.loadmat(str(pairs_file))
    
    # Extract pairs information
    pairs_used = pairs_mat['pairs_used']  # Shape: (n_pairs, 2)
    
    pairs_info = []
    for i in range(len(pairs_used)):
        bipolar_ch = f"b{i+1:03d}"
        pair = pairs_used[i]
        pairs_info.append({
            'bipolar_ch': bipolar_ch,
            'pair_anchor': int(pair[0]),
            'pair_ref': int(pair[1])
        })
    
    pairs_df = pd.DataFrame(pairs_info)
    
    # Create bipolar channel labels using the core library function
    bipolar_labels_df = create_bipolar_channel_labels(pairs_df)
    
    # Save
    bipolar_labels_file = trial_bipolar_dir / "bipolar_channel_labels.csv"
    bipolar_labels_df.to_csv(bipolar_labels_file, index=False)
    
    print(f"\nCreated bipolar labels mapping CSV: {bipolar_labels_file}")
    print(f"Total bipolar channels: {len(bipolar_labels_df)}")
    
    # Print distribution
    print(f"\nRegion type distribution:")
    print(bipolar_labels_df['region_type'].value_counts())
    
    print("\nDone!")


if __name__ == '__main__':
    main()
