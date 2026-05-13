#!/usr/bin/env python3
"""
Unified Session Loading Script

Load raw LFP data from .dat files and save to .mat format.
Supports single trials, batch loading, and smart validation.

Usage Examples:
    # Single trial
    python scripts/preprocessing/load_sessions.py --session 46 --trial 4
    
    # Multiple trials (batch)
    python scripts/preprocessing/load_sessions.py --batch \
        --sessions "73:2,4" "74:2,4" "75:2,4"
    
    # Smart loading with length validation and fallback
    python scripts/preprocessing/load_sessions.py --smart \
        --sessions "73:2,4" "74:2,4" --fallback
    
    # Scan all sessions and load valid trials
    python scripts/preprocessing/load_sessions.py --scan-all \
        --session-range 1 160 --trial-range 1 30

Format for --sessions: "session_num:pre_trial,post_trial"
"""

import sys
from pathlib import Path

# Add ripple_core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'packages' / 'ripple_core'))

import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
import pandas as pd
from datetime import datetime

from ripple_core.load import load_movie_database, load_electrodes


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_OUTPUT_DIR = '/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen'
MIN_DURATION_MIN = 4.0
MAX_DURATION_MIN = 6.5
MAX_MISSING_CHANNELS = 10
TOTAL_CHANNELS = 220


# ============================================================================
# Core Loading Functions
# ============================================================================
def load_single_channel(session_idx: int, trial: int, channel: int) -> np.ndarray:
    """Load LFP data for a single channel."""
    return load_electrodes(session_idx, trial, channel)


def save_channel_to_mat(output_path: Path, lfp_data: np.ndarray,
                        session_number: int, trial: int, channel: int, date: str):
    """Save LFP data to .mat file."""
    mat_data = {
        'lfp': lfp_data,
        'fs': 1000,
        'session': session_number,
        'trial': trial,
        'channel': channel,
        'date': date
    }
    savemat(output_path, mat_data)


def parse_channel_range(channel_str: str) -> list:
    """Parse channel specification: "1-10", "1,5,10", "1-5,10-15"."""
    channels = []
    for part in channel_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            channels.extend(range(int(start), int(end) + 1))
        else:
            channels.append(int(part))
    return sorted(set(channels))


# ============================================================================
# Trial Validation Functions
# ============================================================================
def check_trial_duration(session_idx: int, trial: int) -> tuple:
    """
    Check if trial duration is within acceptable range.
    Returns: (is_valid, duration_minutes, reason)
    """
    try:
        lfp = load_electrodes(session_idx, trial, 1)
        duration_sec = len(lfp) / 1000
        duration_min = duration_sec / 60
        
        if duration_min < MIN_DURATION_MIN:
            return False, duration_min, "too_short"
        elif duration_min > MAX_DURATION_MIN:
            return False, duration_min, "too_long"
        else:
            return True, duration_min, "ok"
    except Exception as e:
        return False, 0.0, f"error: {str(e)[:50]}"


def check_channel_availability(session_idx: int, trial: int) -> tuple:
    """
    Check how many channels are available.
    Returns: (is_valid, channels_available, channels_missing, available_list)
    """
    available = []
    missing = 0
    
    for ch in range(1, TOTAL_CHANNELS + 1):
        try:
            lfp = load_electrodes(session_idx, trial, ch)
            if len(lfp) > 0:
                available.append(ch)
            else:
                missing += 1
        except:
            missing += 1
    
    is_valid = missing <= MAX_MISSING_CHANNELS
    return is_valid, len(available), missing, available


# ============================================================================
# Main Loading Functions
# ============================================================================
def load_trial(session_idx: int, trial: int, base_dir: Path,
               channels_to_load: list = None, force: bool = False) -> dict:
    """
    Load all channels for a single trial.
    
    Returns dict with: success, loaded, skipped, failed, output_dir
    """
    # Get session info
    sessions = load_movie_database()
    if session_idx < 0 or session_idx >= len(sessions):
        raise ValueError(f"Session index {session_idx} out of range (0-{len(sessions)-1})")

    sess_info = sessions[session_idx]
    sess_date = sess_info['date']
    sess_number = sess_info['session']
    
    trial_str = f"{trial:03d}"
    
    # Default channels
    if channels_to_load is None:
        channels_to_load = list(range(1, TOTAL_CHANNELS + 1))
    
    # Create directory structure
    session_output_dir = base_dir / f"session{sess_number:03d}"
    trial_output_dir = session_output_dir / f"trial{trial_str}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    loaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for ch in tqdm(channels_to_load, desc=f"Loading sess{sess_number} trial{trial_str}", leave=False):
        output_chan_dir = trial_output_dir / f"chan{ch:03d}"
        output_chan_dir.mkdir(exist_ok=True)
        
        mat_file_name = f"sess{sess_number:03d}_trial{trial_str}_chan{ch:03d}.mat"
        mat_file_path = output_chan_dir / mat_file_name

        if mat_file_path.exists() and not force:
            skipped_count += 1
            continue

        try:
            lfp_data = load_single_channel(session_idx, trial, ch)
            save_channel_to_mat(mat_file_path, lfp_data, sess_number, trial, ch, sess_date)
            loaded_count += 1
        except Exception as e:
            failed_count += 1
    
    return {
        'success': failed_count == 0,
        'loaded': loaded_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'output_dir': trial_output_dir,
        'session': sess_number,
        'trial': trial
    }


def load_trial_smart(sess_num: int, sess_idx: int, trial_num: int,
                     base_dir: Path, allow_fallback: bool = True) -> dict:
    """
    Smart loading with validation and fallback.
    
    Returns dict with: success, trial_loaded, duration, reason
    """
    print(f"  Checking trial {trial_num:03d}...")
    valid, duration, reason = check_trial_duration(sess_idx, trial_num)
    
    if valid:
        print(f"    ✓ Valid: {duration:.2f} min")
        trial_to_load = trial_num
    else:
        print(f"    ✗ Invalid: {duration:.2f} min ({reason})")
        
        if not allow_fallback:
            return {'success': False, 'trial_loaded': None, 'duration': duration, 'reason': reason}
        
        # Try fallback (trial - 1)
        fallback_trial = trial_num - 1
        if fallback_trial < 1:
            return {'success': False, 'trial_loaded': None, 'duration': 0, 'reason': 'no_fallback'}
        
        print(f"  Checking fallback trial {fallback_trial:03d}...")
        valid_fb, duration_fb, reason_fb = check_trial_duration(sess_idx, fallback_trial)
        
        if valid_fb:
            print(f"    ✓ Fallback valid: {duration_fb:.2f} min")
            trial_to_load = fallback_trial
            duration = duration_fb
        else:
            print(f"    ✗ Fallback invalid: {duration_fb:.2f} min ({reason_fb})")
            return {'success': False, 'trial_loaded': None, 'duration': 0, 'reason': 'fallback_failed'}
    
    # Load the trial
    result = load_trial(sess_idx, trial_to_load, base_dir)
    
    return {
        'success': result['success'],
        'trial_loaded': trial_to_load,
        'trial_requested': trial_num,
        'duration': duration,
        'reason': 'ok' if result['success'] else 'load_failed',
        **result
    }


# ============================================================================
# Batch Processing
# ============================================================================
def parse_session_spec(spec: str) -> tuple:
    """
    Parse session specification: "session_num:trial1,trial2" or "session_num:pre,post"
    Returns: (session_num, [trials])
    """
    parts = spec.split(':')
    session_num = int(parts[0])
    trials = [int(t) for t in parts[1].split(',')]
    return session_num, trials


def load_batch(session_specs: list, base_dir: Path, smart: bool = False,
               allow_fallback: bool = True) -> list:
    """
    Load multiple sessions/trials.
    
    Args:
        session_specs: List of "session:trial1,trial2" strings
        base_dir: Output directory
        smart: Use smart loading with validation
        allow_fallback: Allow fallback to trial-1 (only with smart)
    
    Returns: List of results
    """
    results = []
    
    for spec in session_specs:
        sess_num, trials = parse_session_spec(spec)
        # Session index is typically sess_num - 1, but may vary
        sess_idx = sess_num - 1  # Adjust if needed
        
        print(f"\n{'#'*60}")
        print(f"# SESSION {sess_num}")
        print(f"{'#'*60}")
        
        for trial in trials:
            if smart:
                result = load_trial_smart(sess_num, sess_idx, trial, base_dir, allow_fallback)
            else:
                result = load_trial(sess_idx, trial, base_dir)
            
            result['session'] = sess_num
            result['trial_requested'] = trial
            results.append(result)
    
    return results


def scan_and_load_all(session_range: tuple, trial_range: tuple,
                      base_dir: Path, dry_run: bool = False) -> list:
    """
    Scan all sessions/trials and load valid ones.
    
    Args:
        session_range: (start, end) inclusive
        trial_range: (start, end) inclusive
        base_dir: Output directory
        dry_run: Only report, don't load
    
    Returns: List of results
    """
    results = []
    stats = {
        'total_checked': 0,
        'valid': 0,
        'loaded': 0,
        'skipped_duration': 0,
        'skipped_channels': 0,
        'skipped_exists': 0,
        'failed': 0,
    }
    
    print('='*80)
    print('SCANNING ALL SESSIONS FOR VALID TRIALS')
    print('='*80)
    print(f'Duration criteria: {MIN_DURATION_MIN}-{MAX_DURATION_MIN} minutes')
    print(f'Max missing channels: {MAX_MISSING_CHANNELS}/{TOTAL_CHANNELS}')
    print(f'Session range: {session_range[0]}-{session_range[1]}')
    print(f'Trial range: {trial_range[0]}-{trial_range[1]}')
    print(f'Mode: {"DRY RUN" if dry_run else "LOADING"}')
    print('='*80)
    
    for session_idx in range(session_range[0], session_range[1] + 1):
        print(f'\n[SESSION {session_idx:03d}]')
        
        for trial in range(trial_range[0], trial_range[1] + 1):
            stats['total_checked'] += 1
            
            # Check if already loaded
            trial_dir = base_dir / f'session{session_idx:03d}' / f'trial{trial:03d}'
            if trial_dir.exists():
                chan_dirs = list(trial_dir.glob('chan???'))
                if len(chan_dirs) >= 200:
                    stats['skipped_exists'] += 1
                    continue
            
            # Check duration
            duration_ok, duration, reason = check_trial_duration(session_idx, trial)
            if not duration_ok:
                if "error" not in reason:
                    stats['skipped_duration'] += 1
                continue
            
            # Check channels (slow, only for valid duration)
            print(f'  Trial {trial:03d}: {duration:.2f}min, checking channels...')
            ch_ok, ch_avail, ch_miss, available = check_channel_availability(session_idx, trial)
            
            if not ch_ok:
                print(f'    ✗ Too many missing ({ch_miss})')
                stats['skipped_channels'] += 1
                continue
            
            stats['valid'] += 1
            print(f'    ✓ Valid ({ch_avail} channels)')
            
            if not dry_run:
                result = load_trial(session_idx, trial, base_dir, available)
                if result['success']:
                    stats['loaded'] += 1
                else:
                    stats['failed'] += 1
                results.append(result)
    
    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    print(f"Checked: {stats['total_checked']}")
    print(f"Valid: {stats['valid']}")
    print(f"Loaded: {stats['loaded']}")
    print(f"Skipped (exists): {stats['skipped_exists']}")
    print(f"Skipped (duration): {stats['skipped_duration']}")
    print(f"Skipped (channels): {stats['skipped_channels']}")
    print(f"Failed: {stats['failed']}")
    
    return results


# ============================================================================
# Main CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified session loading script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single trial (using session index)
    python load_sessions.py --session-idx 46 --trial 4
    
    # Batch loading
    python load_sessions.py --batch --sessions "73:2,4" "74:2,4" "75:2,4"
    
    # Smart loading with validation and fallback
    python load_sessions.py --smart --sessions "77:2,5" "78:2,4" --fallback
    
    # Scan all sessions
    python load_sessions.py --scan-all --session-range 1 160 --dry-run
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--batch', action='store_true',
                           help='Batch load specified sessions')
    mode_group.add_argument('--smart', action='store_true',
                           help='Smart loading with validation')
    mode_group.add_argument('--scan-all', action='store_true',
                           help='Scan and load all valid sessions')
    
    # Single trial options
    parser.add_argument('--session-idx', type=int,
                       help='Session index (0-based) for single trial')
    parser.add_argument('--trial', type=int,
                       help='Trial number for single trial')
    parser.add_argument('--channels', type=str, default='1-220',
                       help='Channels to load (e.g., "1-220", "1,5,10")')
    
    # Batch options
    parser.add_argument('--sessions', nargs='+',
                       help='Session specs: "session:trial1,trial2"')
    parser.add_argument('--fallback', action='store_true',
                       help='Allow fallback to trial-1 if validation fails')
    
    # Scan options
    parser.add_argument('--session-range', type=int, nargs=2, default=[1, 160],
                       help='Session range [start end]')
    parser.add_argument('--trial-range', type=int, nargs=2, default=[1, 30],
                       help='Trial range [start end]')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only scan, do not load')
    
    # Common options
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files')
    
    args = parser.parse_args()
    base_dir = Path(args.output_dir)
    
    print("="*70)
    print("UNIFIED SESSION LOADING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Determine mode
    if args.scan_all:
        results = scan_and_load_all(
            tuple(args.session_range),
            tuple(args.trial_range),
            base_dir,
            args.dry_run
        )
    
    elif args.batch or args.smart:
        if not args.sessions:
            parser.error("--batch or --smart requires --sessions")
        
        results = load_batch(
            args.sessions,
            base_dir,
            smart=args.smart,
            allow_fallback=args.fallback
        )
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH LOADING SUMMARY")
        print("="*70)
        successful = sum(1 for r in results if r.get('success', False))
        print(f"Successful: {successful}/{len(results)}")
        
        for r in results:
            status = "✓" if r.get('success') else "✗"
            trial_info = f"{r.get('trial_requested', r.get('trial')):03d}"
            if 'trial_loaded' in r and r['trial_loaded'] != r.get('trial_requested'):
                trial_info += f"→{r['trial_loaded']:03d}"
            print(f"  {status} Session {r['session']:03d} Trial {trial_info}")
    
    elif args.session_idx is not None and args.trial is not None:
        # Single trial mode
        channels = parse_channel_range(args.channels)
    
        # Check duration first
        valid, duration, reason = check_trial_duration(args.session_idx, args.trial)
        if not valid and duration > MAX_DURATION_MIN:
            print(f"⚠️  Trial too long: {duration:.2f} min (>{MAX_DURATION_MIN} min)")
            print("Loading stopped. Use --force to override.")
            if not args.force:
                return
        
        result = load_trial(args.session_idx, args.trial, base_dir, channels, args.force)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Loaded: {result['loaded']}")
        print(f"⊘ Skipped: {result['skipped']}")
        print(f"✗ Failed: {result['failed']}")
        print(f"Output: {result['output_dir']}")
    
    else:
        parser.print_help()
        return
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
