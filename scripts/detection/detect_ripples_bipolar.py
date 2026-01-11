#!/usr/bin/env python3
"""
Script to detect ripples on bipolar-referenced channels.

Usage:
    python scripts/detect_ripples_bipolar.py <trial_bipolar_dir> [--fs FS] [--rp-band LOW HIGH] [--z-low Z] [--z-outlier Z] [--min-dur-ms MS] [--merge-dur-ms MS] [--epoch-ms MS]
    
Example:
    python scripts/detect_ripples_bipolar.py /vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001_bipolar
"""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "ripple_core"))

from ripple_core.signal.bipolar import detect_ripples_on_bipolar_channels
import argparse

def main():
    parser = argparse.ArgumentParser(description='Detect ripples on bipolar-referenced channels')
    parser.add_argument('--session', type=int, required=True, help='Session number (e.g., 1, 46, 134)')
    parser.add_argument('--trial', type=int, required=True, help='Trial number (e.g., 1, 4, 6)')
    parser.add_argument('--fs', type=int, default=1000, help='Sampling frequency (Hz), default 1000')
    parser.add_argument('--rp-band', nargs=2, type=float, default=[100, 140], help='Ripple frequency band (Hz), default 100 140')
    parser.add_argument('--z-low', type=float, default=2.5, help='Z-score threshold, default 2.5')
    parser.add_argument('--z-outlier', type=float, default=9.0, help='Z-score for outlier clipping, default 9.0')
    parser.add_argument('--min-dur-ms', type=float, default=30, help='Minimum ripple duration (ms), default 30')
    parser.add_argument('--merge-dur-ms', type=float, default=10, help='Merge ripples closer than this (ms), default 10')
    parser.add_argument('--epoch-ms', type=float, default=200, help='Epoch window around peaks (ms), default 200')
    
    args = parser.parse_args()
    
    # Construct trial bipolar directory
    base_dir = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
    trial_bipolar_dir = base_dir / f"session{args.session:03d}" / f"trial{args.trial:03d}_bipolar"
    
    # Get parameters from args
    fs = args.fs
    rp_band = tuple(args.rp_band)
    z_low = args.z_low
    z_outlier = args.z_outlier
    min_dur_ms = args.min_dur_ms
    merge_dur_ms = args.merge_dur_ms
    epoch_ms = args.epoch_ms
    
    print("=" * 70)
    print("Ripple Detection on Bipolar Channels")
    print("=" * 70)
    print(f"Bipolar directory: {trial_bipolar_dir}")
    print(f"Detection parameters:")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Ripple band: {rp_band[0]}-{rp_band[1]} Hz")
    print(f"  Z-score threshold: {z_low}")
    print(f"  Z-score outlier: {z_outlier}")
    print(f"  Min duration: {min_dur_ms} ms")
    print(f"  Merge duration: {merge_dur_ms} ms")
    print(f"  Epoch window: {epoch_ms} ms")
    print()
    
    if not trial_bipolar_dir.is_dir():
        print(f"ERROR: Bipolar directory not found: {trial_bipolar_dir}")
        sys.exit(1)
    
    try:
        result = detect_ripples_on_bipolar_channels(
            trial_bipolar_dir=trial_bipolar_dir,
            fs=fs,
            rp_band=rp_band,
            z_low=z_low,
            z_outlier=z_outlier,
            min_dur_ms=min_dur_ms,
            merge_dur_ms=merge_dur_ms,
            epoch_ms=epoch_ms,
        )
        
        print("\n" + "=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"Summary file: {result['summary_file']}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
