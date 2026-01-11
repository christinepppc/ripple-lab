#!/usr/bin/env python3
"""
Script to process bipolar re-referencing for a trial.

Usage:
    python scripts/process_bipolar_trial.py [trial_dir] [--bad-channels CH1 CH2 ...]
    
Example:
    python scripts/process_bipolar_trial.py /vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001
"""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "ripple_core"))

from ripple_core.signal.bipolar import process_bipolar_referencing
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process bipolar re-referencing for a trial')
    parser.add_argument('--session', type=int, required=True, help='Session number (e.g., 1, 46, 134)')
    parser.add_argument('--trial', type=int, required=True, help='Trial number (e.g., 1, 4, 6)')
    parser.add_argument('--bad-channels', nargs='*', type=int, default=[], help='Bad channel numbers to exclude')
    
    args = parser.parse_args()
    
    # Construct trial directory
    base_dir = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    trial_dir = base_dir / f"session{args.session:03d}" / f"trial{args.trial:03d}"
    
    bad_channels = args.bad_channels
    
    print("=" * 70)
    print("Bipolar Re-referencing Processing")
    print("=" * 70)
    print(f"Trial directory: {trial_dir}")
    print(f"Bad channels: {bad_channels if bad_channels else 'None'}")
    print()
    
    if not trial_dir.is_dir():
        print(f"ERROR: Trial directory not found: {trial_dir}")
        sys.exit(1)
    
    try:
        print("Processing bipolar pairs...")
        result = process_bipolar_referencing(
            trial_dir=trial_dir,
            bad_channels=bad_channels,
            prefer="horizontal"
        )
        
        print()
        print("=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"Output directory: {result['output_dir']}")
        print(f"Total pairs processed: {result['n_pairs_processed']}")
        print(f"  - Horizontal pairs: {result['n_horizontal']}")
        print(f"  - Vertical pairs: {result['n_vertical']}")
        print()
        print(f"Bipolar channels saved as: b001, b002, ..., b{result['n_pairs_processed']:03d}")
        print(f"Summary saved to: {result['output_dir'] / 'pairs_used.mat'}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
