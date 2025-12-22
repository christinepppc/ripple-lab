#!/usr/bin/env python3
"""
CAR Re-referencing Processing Script

This script processes neural data using Common Average Reference (CAR) re-referencing.
Each channel is referenced to the average of all other channels within the same bank.

Usage:
    python scripts/car/process_car.py /path/to/trial/dir --bad-channels 1,2,3
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from ripple_core.ripple_core.signal.car import process_car_referencing


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process CAR re-referencing")
    parser.add_argument("trial_dir", help="Trial directory containing chan### subdirectories")
    parser.add_argument("--bad-channels", "-b", help="Bad channels (comma-separated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Parse bad channels
    bad_channels = []
    if args.bad_channels:
        bad_channels = [int(ch.strip()) for ch in args.bad_channels.split(",")]
    
    try:
        summary = process_car_referencing(
            root_dir=args.trial_dir,
            bad_channels=bad_channels
        )
        
        print(f"\n✓ CAR processing completed successfully!")
        print(f"Results saved to: {summary['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


