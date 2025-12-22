#!/usr/bin/env python3
"""
Visualize CAR re-referenced ripple data.

This script creates comprehensive visualizations for CAR re-referenced data,
mirroring the MATLAB visualizeRipples.m functionality.

Usage:
    python scripts/car/visualize_car_ripples.py /path/to/car_processed --max-channels-per-bank 5
"""

import argparse
import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from ripple_core.visualize_reref import visualize_car_ripples


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize CAR ripple data")
    parser.add_argument("car_dir", help="Directory containing CAR-processed data")
    parser.add_argument("--output", "-o", help="Output directory (default: car_dir + '_visualizations')")
    parser.add_argument("--max-channels-per-bank", "-m", type=int, default=5, 
                       help="Maximum number of channels per bank to visualize (default: 5)")
    parser.add_argument("--window-ms", "-w", type=int, default=200,
                       help="Window size in milliseconds (default: 200)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        car_path = Path(args.car_dir)
        output_dir = str(car_path.parent / (car_path.name + "_visualizations"))
    
    try:
        visualize_car_ripples(
            car_dir=args.car_dir,
            output_dir=output_dir,
            max_channels_per_bank=args.max_channels_per_bank,
            window_ms=args.window_ms
        )
        
        print(f"\n✓ CAR ripple visualizations completed!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


