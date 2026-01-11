#!/usr/bin/env python3
"""
Load raw LFP data from .dat files and save to .mat format.

Usage:
    python scripts/load_sessions.py --session_idx 46 --trial 4 --channels 1-220
    python scripts/load_sessions.py --session_idx 46 --trial 4 --channels 1,2,3,10
    python scripts/load_sessions.py --session_idx 46 --trial 4 --channels 1-10 --force
"""

import sys
from pathlib import Path

# Add ripple_core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'packages' / 'ripple_core'))

import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

from ripple_core.load import load_movie_database, load_electrodes


def load_single_channel(session_idx: int, trial: int, channel: int) -> np.ndarray:
    """
    Load LFP data for a single channel.
    
    Parameters
    ----------
    session_idx : int
        Session index (0-based) in the movie database
    trial : int
        Trial number (1-based)
    channel : int
        Channel number (1-based)
    
    Returns
    -------
    lfp_data : np.ndarray
        LFP data for the channel (1D array)
    """
    return load_electrodes(session_idx, trial, channel)


def save_channel_to_mat(
    output_path: Path,
    lfp_data: np.ndarray,
    session_number: int,
    trial: int,
    channel: int,
    date: str
):
    """
    Save LFP data to .mat file.
    
    Parameters
    ----------
    output_path : Path
        Path to save the .mat file
    lfp_data : np.ndarray
        LFP data to save
    session_number : int
        Session number (for metadata)
    trial : int
        Trial number (for metadata)
    channel : int
        Channel number (for metadata)
    date : str
        Session date (for metadata)
    """
    mat_data = {
        'lfp': lfp_data,
        'fs': 1000,  # Sampling rate in Hz
        'session': session_number,
        'trial': trial,
        'channel': channel,
        'date': date
    }
    savemat(output_path, mat_data)


def load_and_save_session_trial(
    session_idx: int,
    trial: int,
    base_dir: Path,
    channels_to_load: list,
    force_overwrite: bool = False
):
    """
    Load and save all specified channels for a session/trial.
    
    Parameters
    ----------
    session_idx : int
        Session index (0-based) in the movie database
    trial : int
        Trial number (1-based)
    base_dir : Path
        Base directory for saving data
    channels_to_load : list of int
        List of channel numbers to load (1-based)
    force_overwrite : bool
        If True, overwrite existing files
    """
    # Get session info
    sessions = load_movie_database()
    if session_idx < 0 or session_idx >= len(sessions):
        raise ValueError(f"Session index {session_idx} out of range (0-{len(sessions)-1})")

    sess_info = sessions[session_idx]
    sess_date = sess_info['date']
    sess_number = sess_info['session']  # 1-based session number

    trial_str = str(trial).zfill(3)
    
    # Check trial duration before loading (max 6 minutes)
    print(f"Checking trial duration for session {sess_number}, trial {trial}...")
    try:
        # Load first channel to check duration
        test_data = load_single_channel(session_idx, trial, 1)
        duration_seconds = len(test_data) / 1000  # 1000 Hz sampling rate
        duration_minutes = duration_seconds / 60
        
        print(f"  Duration: {duration_seconds:.1f} seconds = {duration_minutes:.2f} minutes")
        
        if duration_minutes > 6.5:
            print("\n" + "="*80)
            print("⚠️  TRIAL TOO LONG")
            print("="*80)
            print(f"Trial {trial} is {duration_minutes:.2f} minutes long (> 6.5 minutes threshold)")
            print("Loading stopped to prevent excessive processing time.")
            print("="*80)
            return
        else:
            print(f"  ✓ Duration OK (under 6 minutes)\n")
    except Exception as e:
        print(f"  ✗ Could not check duration: {e}")
        print("  Proceeding with loading...\n")
    
    # Create directory structure: base/sessionXXX/trialYYY/
    session_output_dir = base_dir / f"session{sess_number}"
    trial_output_dir = session_output_dir / f"trial{trial_str}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base directory:    {base_dir}")
    print(f"Session directory: {session_output_dir}")
    print(f"Trial directory:   {trial_output_dir}")
    print(f"Processing {len(channels_to_load)} channels...\n")

    loaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for ch in tqdm(channels_to_load, desc="Loading channels"):
        # Create channel subdirectory
        output_chan_dir = trial_output_dir / f"chan{ch:03d}"
        output_chan_dir.mkdir(exist_ok=True)
        
        # Output filename: sess###_trial###_chan###.mat
        mat_file_name = f"sess{sess_number}_trial{trial_str}_chan{ch:03d}.mat"
        mat_file_path = output_chan_dir / mat_file_name

        # Skip if file exists and not forcing overwrite
        if mat_file_path.exists() and not force_overwrite:
            skipped_count += 1
            continue

        try:
            lfp_data = load_single_channel(session_idx, trial, ch)
            save_channel_to_mat(mat_file_path, lfp_data, sess_number, trial, ch, sess_date)
            loaded_count += 1
        except Exception as e:
            print(f"\n  ✗ Error loading/saving channel {ch}: {e}")
            failed_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Successfully loaded:  {loaded_count} channels")
    if skipped_count > 0:
        print(f"⊘ Skipped (existing):   {skipped_count} channels")
    if failed_count > 0:
        print(f"✗ Failed:               {failed_count} channels")
    print(f"Data saved to: {trial_output_dir}/")
    print("="*80)


def parse_channel_range(channel_str: str) -> list:
    """
    Parse channel specification string.
    
    Examples:
        "1-10" -> [1, 2, 3, ..., 10]
        "1,5,10" -> [1, 5, 10]
        "1-5,10-15" -> [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    """
    channels = []
    for part in channel_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            channels.extend(range(int(start), int(end) + 1))
        else:
            channels.append(int(part))
    return sorted(set(channels))


def main():
    parser = argparse.ArgumentParser(
        description="Load raw LFP data and save to .mat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Load all channels (1-220) for session 46, trial 4:
        python scripts/load_sessions.py --session_idx 46 --trial 4 --channels 1-220
    
    Load specific channels:
        python scripts/load_sessions.py --session_idx 46 --trial 4 --channels 1,2,3,10
    
    Load a range with force overwrite:
        python scripts/load_sessions.py --session_idx 46 --trial 4 --channels 1-10 --force
        """
    )
    
    parser.add_argument(
        '--session_idx',
        type=int,
        required=True,
        help='Session index (0-based) in the movie database'
    )
    
    parser.add_argument(
        '--trial',
        type=int,
        required=True,
        help='Trial number (1-based, e.g., 4 for trial004)'
    )
    
    parser.add_argument(
        '--channels',
        type=str,
        default='1-220',
        help='Channels to load (e.g., "1-220", "1,5,10", "1-10,50-60")'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen',
        help='Base output directory'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing files'
    )
    
    args = parser.parse_args()
    
    # Parse channel specification
    channels_to_load = parse_channel_range(args.channels)
    
    # Validate channels
    if not channels_to_load:
        raise ValueError("No valid channels specified")
    if min(channels_to_load) < 1 or max(channels_to_load) > 220:
        raise ValueError("Channel numbers must be between 1 and 220")
    
    # Convert to Path
    base_dir = Path(args.output_dir)
    
    # Load and save
    load_and_save_session_trial(
        session_idx=args.session_idx,
        trial=args.trial,
        base_dir=base_dir,
        channels_to_load=channels_to_load,
        force_overwrite=args.force
    )


if __name__ == '__main__':
    main()
