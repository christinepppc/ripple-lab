#!/usr/bin/env python3
"""
Command-line script for bipolar referencing processing.

Usage:
    python scripts/process_bipolar.py /path/to/trial/dir --output /path/to/output --bad-channels 1,2,3
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from ripple_core.signal import create_bipolar_processing_script
from ripple_core.signal.pairs import WHITE_MATTER_CHANNELS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process bipolar referencing for neural data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python scripts/process_bipolar.py /vol/brains/bd3/peasaranlab/Archie_RecStim_vSUBNETS220_2nd/180531/001

  # With custom output directory
  python scripts/process_bipolar.py /vol/brains/bd3/peasaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001 --output /path/to/output

  # With bad channels specified
  python scripts/process_bipolar.py /vol/brains/bd3/Archie_RecStim_vSUBNETS220_2nd/180531/001 --bad-channels 1,2,3

  # Full example
  python scripts/process_bipolar.py /vol/brains/bd3/Archie_RecStim_vSUBNETS220_2nd/180531/001 --output /path/to/output --bad-channels 1,2,3 --prefer vertical
        """
    )
    
    parser.add_argument(
        "trial_dir",
        type=str,
        help="Path to trial directory containing chan###/lfp.mat files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for processed results (default: trial_dir + '_bipolar_processed')"
    )
    
    parser.add_argument(
        "--bad-channels", "-b",
        type=str,
        help="Comma-separated list of bad channel IDs to exclude (e.g., '1,2,3')"
    )
    
    parser.add_argument(
        "--prefer",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Which pairs to process first (default: horizontal)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate trial directory
    trial_dir = Path(args.trial_dir)
    if not trial_dir.exists():
        print(f"Error: Trial directory not found: {trial_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = trial_dir.parent / f"{trial_dir.name}_bipolar_processed"
    
    # Parse bad channels
    bad_channels = []
    if args.bad_channels:
        try:
            bad_channels = [int(ch.strip()) for ch in args.bad_channels.split(",")]
        except ValueError:
            print(f"Error: Invalid bad channels format: {args.bad_channels}")
            print("Use comma-separated integers (e.g., '1,2,3')")
            sys.exit(1)
    
    # Print configuration
    print("Bipolar Referencing Processing")
    print("=" * 40)
    print(f"Trial directory: {trial_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bad channels: {bad_channels if bad_channels else 'None'}")
    print(f"White matter channels excluded: {len(WHITE_MATTER_CHANNELS)} channels")
    print(f"Prefer: {args.prefer}")
    print()
    
    try:
        # Process bipolar referencing
        from ripple_core.signal import process_bipolar_referencing, get_bipolar_lfp_matrix
        
        print("Processing bipolar referencing...")
        dataset = process_bipolar_referencing(
            root_dir=trial_dir,
            bad_channels=bad_channels,
            prefer=args.prefer
        )
        
        print(f"✓ Processed {dataset.n_total} bipolar channels")
        print(f"  - Horizontal pairs: {dataset.n_horizontal}")
        print(f"  - Vertical pairs: {dataset.n_vertical}")
        
        # Extract and save LFP matrix
        print("Extracting LFP matrix...")
        lfp_matrix, channel_ids, fs = get_bipolar_lfp_matrix(dataset)
        
        print(f"✓ LFP matrix shape: {lfp_matrix.shape}")
        print(f"✓ Sampling frequency: {fs} Hz")
        print(f"✓ Duration: {lfp_matrix.shape[1] / fs:.2f} seconds")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LFP matrix
        import scipy.io as sio
        lfp_file = output_dir / "bipolar_lfp_matrix.mat"
        sio.savemat(lfp_file, {
            "lfp_matrix": lfp_matrix,
            "channel_ids": np.array(channel_ids),
            "fs": fs,
            "n_samples": lfp_matrix.shape[1],
            "duration_sec": lfp_matrix.shape[1] / fs
        }, do_compression=True)
        
        print(f"✓ LFP matrix saved to: {lfp_file}")
        
        # Save processing summary
        summary_file = output_dir / "processing_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Bipolar Referencing Processing Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Trial directory: {trial_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Bad channels: {bad_channels}\n")
            f.write(f"Prefer: {args.prefer}\n")
            f.write(f"Total bipolar channels: {dataset.n_total}\n")
            f.write(f"Horizontal pairs: {dataset.n_horizontal}\n")
            f.write(f"Vertical pairs: {dataset.n_vertical}\n")
            f.write(f"LFP matrix shape: {lfp_matrix.shape}\n")
            f.write(f"Sampling frequency: {fs} Hz\n")
            f.write(f"Duration: {lfp_matrix.shape[1] / fs:.2f} seconds\n")
        
        print(f"✓ Processing summary saved to: {summary_file}")
        
        print("\n✓ Bipolar processing completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
