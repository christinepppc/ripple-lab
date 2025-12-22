#!/usr/bin/env python3
"""
Batch bipolar processing script for multiple sessions and trials.

This script processes bipolar referencing for multiple sessions and trials,
excluding white matter channels as specified.

Usage:
    python scripts/bipolar/batch_process_bipolar.py --base-dir /path/to/data --sessions 134,135,136 --trials 001,002,003
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from ripple_core.ripple_core.signal import process_bipolar_referencing, get_bipolar_lfp_matrix
from ripple_core.ripple_core.signal.pairs import WHITE_MATTER_CHANNELS


def find_available_sessions(base_dir: Path) -> list:
    """Find all available sessions in the base directory."""
    sessions = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("session"):
            try:
                session_num = int(item.name.replace("session", ""))
                sessions.append(session_num)
            except ValueError:
                continue
    return sorted(sessions)


def find_available_trials(session_dir: Path) -> list:
    """Find all available trials in a session directory."""
    trials = []
    for item in session_dir.iterdir():
        if item.is_dir() and item.name.startswith("trial"):
            try:
                trial_num = int(item.name.replace("trial", ""))
                trials.append(trial_num)
            except ValueError:
                continue
    return sorted(trials)


def check_trial_structure(trial_dir: Path) -> bool:
    """Check if a trial directory has the expected structure with chan### directories."""
    if not trial_dir.exists():
        return False
    
    # Check for chan directories
    chan_dirs = list(trial_dir.glob("chan*"))
    if len(chan_dirs) < 100:  # Should have around 220 channels
        return False
    
    # Check if at least some chan directories have .mat files
    mat_files_found = 0
    for chan_dir in chan_dirs[:10]:  # Check first 10 channels
        if chan_dir.is_dir():
            mat_files = list(chan_dir.glob("*.mat"))
            if mat_files:
                mat_files_found += 1
    
    return mat_files_found > 5  # At least 5 channels should have .mat files


def process_single_trial(trial_dir: Path, output_base: Path, bad_channels: list = None) -> dict:
    """Process a single trial for bipolar referencing."""
    trial_name = trial_dir.name
    session_name = trial_dir.parent.name
    
    print(f"Processing {session_name}/{trial_name}...")
    
    # Create output directory
    output_dir = output_base / f"{session_name}_{trial_name}_bipolar_processed"
    
    try:
        # Process bipolar referencing
        dataset = process_bipolar_referencing(
            root_dir=trial_dir,
            bad_channels=bad_channels,
            prefer="horizontal"
        )
        
        # Extract LFP matrix
        lfp_matrix, channel_ids, fs = get_bipolar_lfp_matrix(dataset)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LFP matrix
        import scipy.io as sio
        lfp_file = output_dir / "bipolar_lfp_matrix.mat"
        sio.savemat(lfp_file, {
            "lfp_matrix": lfp_matrix,
            "channel_ids": channel_ids,
            "fs": fs,
            "n_samples": lfp_matrix.shape[1],
            "duration_sec": lfp_matrix.shape[1] / fs,
            "white_matter_excluded": list(WHITE_MATTER_CHANNELS)
        }, do_compression=True)
        
        # Save processing summary
        summary = {
            "session": session_name,
            "trial": trial_name,
            "input_dir": str(trial_dir),
            "output_dir": str(output_dir),
            "n_horizontal": dataset.n_horizontal,
            "n_vertical": dataset.n_vertical,
            "n_total": dataset.n_total,
            "lfp_matrix_shape": lfp_matrix.shape,
            "sampling_frequency": fs,
            "duration_sec": lfp_matrix.shape[1] / fs,
            "white_matter_channels_excluded": len(WHITE_MATTER_CHANNELS),
            "gray_matter_channels_used": len(channel_ids),
            "processing_time": datetime.now().isoformat()
        }
        
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Success: {dataset.n_total} bipolar channels, {lfp_matrix.shape[1]/fs:.1f}s duration")
        return summary
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            "session": session_name,
            "trial": trial_name,
            "error": str(e),
            "processing_time": datetime.now().isoformat()
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch process bipolar referencing for multiple sessions and trials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific sessions and trials
  python scripts/bipolar/batch_process_bipolar.py --base-dir /vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen --sessions 134,135,136 --trials 001,002,003

  # Process all available sessions and trials
  python scripts/bipolar/batch_process_bipolar.py --base-dir /vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen --all

  # Process with custom output directory
  python scripts/bipolar/batch_process_bipolar.py --base-dir /path/to/data --sessions 134 --trials 001 --output /path/to/output
        """
    )
    
    parser.add_argument(
        "--base-dir", "-b",
        type=str,
        required=True,
        help="Base directory containing session###/trial###/chan### structure"
    )
    
    parser.add_argument(
        "--sessions", "-s",
        type=str,
        help="Comma-separated list of session numbers (e.g., '134,135,136')"
    )
    
    parser.add_argument(
        "--trials", "-t",
        type=str,
        help="Comma-separated list of trial numbers (e.g., '001,002,003')"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available sessions and trials"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output base directory (default: base_dir + '_bipolar_batch_processed')"
    )
    
    parser.add_argument(
        "--bad-channels",
        type=str,
        help="Comma-separated list of bad channel IDs to exclude (e.g., '1,2,3')"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate base directory
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = base_dir.parent / f"{base_dir.name}_bipolar_batch_processed"
    
    # Parse bad channels
    bad_channels = []
    if args.bad_channels:
        try:
            bad_channels = [int(ch.strip()) for ch in args.bad_channels.split(",")]
        except ValueError:
            print(f"Error: Invalid bad channels format: {args.bad_channels}")
            sys.exit(1)
    
    # Determine sessions to process
    if args.all:
        sessions = find_available_sessions(base_dir)
        if not sessions:
            print("Error: No sessions found in base directory")
            sys.exit(1)
    elif args.sessions:
        try:
            sessions = [int(s.strip()) for s in args.sessions.split(",")]
        except ValueError:
            print(f"Error: Invalid sessions format: {args.sessions}")
            sys.exit(1)
    else:
        print("Error: Must specify either --sessions or --all")
        sys.exit(1)
    
    # Determine trials to process
    if args.trials:
        try:
            trials = [int(t.strip()) for t in args.trials.split(",")]
        except ValueError:
            print(f"Error: Invalid trials format: {args.trials}")
            sys.exit(1)
    else:
        # Find trials from first session
        first_session_dir = base_dir / f"session{sessions[0]:03d}"
        trials = find_available_trials(first_session_dir)
        if not trials:
            print("Error: No trials found")
            sys.exit(1)
    
    # Print configuration
    print("Batch Bipolar Processing")
    print("=" * 50)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_base}")
    print(f"Sessions: {sessions}")
    print(f"Trials: {trials}")
    print(f"Bad channels: {bad_channels if bad_channels else 'None'}")
    print(f"White matter channels excluded: {len(WHITE_MATTER_CHANNELS)}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Find all valid trial directories
    valid_trials = []
    for session in sessions:
        session_dir = base_dir / f"session{session:03d}"
        if not session_dir.exists():
            print(f"Warning: Session directory not found: {session_dir}")
            continue
        
        for trial in trials:
            trial_dir = session_dir / f"trial{trial:03d}"
            if check_trial_structure(trial_dir):
                valid_trials.append(trial_dir)
            else:
                print(f"Warning: Invalid trial structure: {trial_dir}")
    
    if not valid_trials:
        print("Error: No valid trials found")
        sys.exit(1)
    
    print(f"Found {len(valid_trials)} valid trials to process")
    print()
    
    if args.dry_run:
        print("Dry run - would process the following trials:")
        for trial_dir in valid_trials:
            print(f"  {trial_dir}")
        return
    
    # Process trials
    results = []
    start_time = time.time()
    
    for i, trial_dir in enumerate(valid_trials, 1):
        print(f"[{i}/{len(valid_trials)}] ", end="")
        result = process_single_trial(trial_dir, output_base, bad_channels)
        results.append(result)
        print()
    
    # Save batch summary
    batch_summary = {
        "batch_processing_time": datetime.now().isoformat(),
        "total_processing_time_sec": time.time() - start_time,
        "total_trials": len(valid_trials),
        "successful_trials": len([r for r in results if "error" not in r]),
        "failed_trials": len([r for r in results if "error" in r]),
        "white_matter_channels_excluded": list(WHITE_MATTER_CHANNELS),
        "bad_channels": bad_channels,
        "results": results
    }
    
    summary_file = output_base / "batch_processing_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(batch_summary, f, indent=2)
    
    # Print final summary
    print("Batch Processing Complete!")
    print("=" * 50)
    print(f"Total trials processed: {len(valid_trials)}")
    print(f"Successful: {batch_summary['successful_trials']}")
    print(f"Failed: {batch_summary['failed_trials']}")
    print(f"Total processing time: {batch_summary['total_processing_time_sec']:.1f} seconds")
    print(f"Results saved to: {output_base}")
    print(f"Batch summary: {summary_file}")


if __name__ == "__main__":
    main()
