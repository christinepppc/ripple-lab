# ripple_core/signal/bipolar.py
"""
Bipolar re-referencing for neural data.

This module implements bipolar re-referencing where pairs of adjacent channels
within the same bank are subtracted (i - j) to create new bipolar-referenced channels.
"""

from __future__ import annotations
from typing import Iterable, Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import scipy.io as sio

# White matter channels to exclude from bipolar referencing
WHITE_MATTER_CHANNELS = {
    2, 10, 14, 16, 25, 26, 28, 36, 39, 42, 46, 53, 54, 55, 57, 59, 60, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 85, 87, 94, 95, 100, 101, 102, 106, 111, 117, 125, 128, 131, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 163, 164, 165, 167, 168, 171, 173, 174, 175, 176, 178, 180, 182, 184, 192, 198, 199, 200, 201, 202, 204, 207, 208, 210, 212, 215, 216, 218
}

# Exact 26 x 10 grid from MATLAB
_LAYOUT_26x10 = np.array([
    [  2,  1,  4,  3, 98,  0,  0,  0,  0,  0],
    [  6,  5,  8,  7, 97,  0,  0,  0,  0,  0],
    [102,101,108, 10,100, 99,  0,  0,  0,  0],
    [104,103,110,  9, 11,109,  0,  0,  0,  0],
    [106,105,107, 12, 14, 13, 22,  0,  0,  0],
    [ 16, 15, 18, 17, 20, 19, 21,  0,  0,  0],
    [ 24, 23, 27,114, 30, 29,116,115,  0,  0],
    [ 26, 25,111,113, 32, 31,118,117,  0,  0],
    [ 28, 34, 33,120,119,124,126,125,128,  0],
    [112, 36, 35, 38,122,123, 40, 39,127,  0],
    [130,129,134, 42, 37,121, 44, 43, 46, 45],
    [132,131,133, 41, 56, 55,136,135, 59, 62],
    [ 48, 47, 54, 53, 58, 57,138,137,149, 61],
    [ 50, 49,140,142,144, 60,148,150,152, 64],
    [ 52, 51,139,141,143,146,145,147,151, 63],
    [162, 66, 65,161,164,163,154,156,158,160],
    [166, 68, 67,165,168,167,153,155,157,159],
    [170,169,172,171,174,173,175,178, 70,177],
    [180,179,182,181, 72, 74,176, 75, 77, 69],
    [184,183,186,185, 71, 73, 76, 78, 80,191],
    [188,187,190,189, 84, 83, 86, 85, 88,192],
    [194, 79,193,196,195,199, 89, 94, 96, 87],
    [198, 82, 81,197,200, 90, 92, 93, 95,  0],
    [202,201,204,203,206,205, 91,208,  0,  0],
    [207,210,209,212,211,214,213,  0,  0,  0],
    [216,215,218,217,220,219,  0,  0,  0,  0],
], dtype=int)

def _bank_of(ch: int) -> int:
    """Determine which bank an electrode belongs to."""
    if   1  <= ch <=  32: return 1
    elif 33 <= ch <=  64: return 2
    elif 65 <= ch <=  96: return 3
    elif 97 <= ch <= 128: return 4
    elif 129<= ch <= 160: return 5
    elif 161<= ch <= 192: return 6
    elif 193<= ch <= 220: return 7
    return -1

def make_bipolar_pairs_from_grid(badCh: Iterable[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build horizontal and vertical bipolar pairs from a channel-ID grid.
    Mirrors MATLAB's make_bipolar_pairs_from_grid.m
    
    - Only pairs within the SAME bank (headstage)
    - Skips blanks (0) and user-specified bad channels
    - Sign convention: y = x(i) - x(j)
    - Horizontal: left minus right; Vertical: top minus bottom
    
    Args:
        badCh: Iterable of bad channel IDs to exclude
        
    Returns:
        pairs_h: (K_h, 2) array of horizontal pairs [i, j] where j = right neighbor
        pairs_v: (K_v, 2) array of vertical pairs [i, j] where j = below neighbor
    """
    G = _LAYOUT_26x10.copy()
    bad = set(int(b) for b in badCh)
    isBad = np.zeros(221, dtype=bool)
    for ch in bad:
        if 1 <= ch <= 220:
            isBad[ch] = True
    
    R, C = G.shape
    pairs_h: List[tuple[int, int]] = []
    pairs_v: List[tuple[int, int]] = []
    
    # Horizontal pairs: (r,c) - (r,c+1)
    for r in range(R):
        for c in range(C - 1):
            i = int(G[r, c])
            j = int(G[r, c + 1])
            if i == 0 or j == 0:
                continue
            if isBad[i] or isBad[j]:
                continue
            if _bank_of(i) != _bank_of(j):
                continue  # stay within bank
            pairs_h.append((i, j))
    
    # Vertical pairs: (r,c) - (r+1,c)
    for r in range(R - 1):
        for c in range(C):
            i = int(G[r, c])
            j = int(G[r + 1, c])
            if i == 0 or j == 0:
                continue
            if isBad[i] or isBad[j]:
                continue
            if _bank_of(i) != _bank_of(j):
                continue  # stay within bank
            pairs_v.append((i, j))
    
    pairs_h_array = np.asarray(pairs_h, dtype=int) if pairs_h else np.empty((0, 2), dtype=int)
    pairs_v_array = np.asarray(pairs_v, dtype=int) if pairs_v else np.empty((0, 2), dtype=int)
    
    return pairs_h_array, pairs_v_array

def _find_lfp_file(chan_dir: Path) -> Optional[Path]:
    """
    Find the LFP .mat file in a channel directory.
    Pattern: sess###_trial###_chan###_*.mat
    """
    if not chan_dir.is_dir():
        return None
    
    # Look for .mat files matching the pattern
    mat_files = list(chan_dir.glob("sess*_trial*_chan*.mat"))
    if len(mat_files) == 1:
        return mat_files[0]
    
    return None

def _load_lfp_from_mat(mat_path: Path) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Load LFP data from a .mat file.
    
    Returns:
        lfp: 1D array of LFP data (or None if not found)
        fs: Sampling rate (or None if not found)
    """
    try:
        data = sio.loadmat(str(mat_path))
        
        # Look for 'lfp' variable
        if 'lfp' in data:
            lfp = data['lfp']
            # Ensure it's 1D
            if lfp.ndim > 1:
                lfp = lfp.flatten()
            lfp = lfp.astype(np.float64)
        else:
            return None, None
        
        # Look for 'fs' variable
        fs = None
        if 'fs' in data:
            fs_val = data['fs']
            if isinstance(fs_val, np.ndarray):
                fs = float(fs_val.item())
            else:
                fs = float(fs_val)
        
        return lfp, fs
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None, None

def process_bipolar_referencing(
    trial_dir: str | Path,
    bad_channels: Iterable[int] = (),
    prefer: str = "horizontal"
) -> Dict:
    """
    Process bipolar re-referencing for a trial directory.
    
    This function:
    1. Finds all channel directories (chan001, chan002, etc.)
    2. Gets bipolar pairs from make_bipolar_pairs_from_grid
    3. For each pair, loads LFP data and computes i - j
    4. Saves bipolar-referenced channels as b001, b002, etc. in trial_dir_bipolar/
    
    Args:
        trial_dir: Path to trial directory (e.g., .../session134/trial001)
        bad_channels: List of bad channel IDs to exclude
        prefer: "horizontal" or "vertical" - which pairs to process first
        
    Returns:
        Dictionary with summary information:
        {
            'output_dir': Path to output directory,
            'n_pairs_processed': int,
            'n_horizontal': int,
            'n_vertical': int,
            'pairs_used': np.ndarray of pairs that were successfully processed
        }
    """
    trial_path = Path(trial_dir)
    if not trial_path.is_dir():
        raise ValueError(f"Trial directory not found: {trial_path}")
    
    # Create output directory
    output_dir = trial_path.parent / f"{trial_path.name}_bipolar"
    output_dir.mkdir(exist_ok=True)
    
    # Get bipolar pairs
    pairs_h, pairs_v = make_bipolar_pairs_from_grid(bad_channels)
    
    # Determine processing order
    if prefer == "vertical":
        pairs_to_process = [(pairs_v, "vertical"), (pairs_h, "horizontal")]
    else:
        pairs_to_process = [(pairs_h, "horizontal"), (pairs_v, "vertical")]
    
    # Track which electrodes have been used (both i and j get marked as used)
    # Once an electrode is used in any pair, it cannot be used again
    used_electrodes = set()
    pairs_used = []
    bipolar_counter = 1
    
    # Discover available channels
    available_channels = {}
    for ch_dir in sorted(trial_path.glob("chan???")):
        chan_num_str = ch_dir.name.replace("chan", "")
        try:
            chan_num = int(chan_num_str)
            lfp_file = _find_lfp_file(ch_dir)
            if lfp_file is not None:
                available_channels[chan_num] = lfp_file
        except ValueError:
            continue
    
    print(f"Found {len(available_channels)} available channels")
    
    # Process pairs
    for pairs_array, pair_type in pairs_to_process:
        for i, j in pairs_array:
            # Skip if either channel is not available
            if i not in available_channels or j not in available_channels:
                continue
            
            # CRITICAL: Skip if EITHER electrode has already been used in a previous pair
            if i in used_electrodes or j in used_electrodes:
                continue
            
            # CRITICAL: White matter channel handling
            # - If BOTH are white matter → skip the pair
            # - If ONE is white matter → use the non-white matter channel as anchor
            i_is_wm = i in WHITE_MATTER_CHANNELS
            j_is_wm = j in WHITE_MATTER_CHANNELS
            
            if i_is_wm and j_is_wm:
                # Both are white matter → skip
                continue
            
            # Determine which electrode to use as anchor (the one being subtracted FROM)
            # If i is white matter but j is not → use j as anchor (j - i)
            # Otherwise → use i as anchor (i - j)
            if i_is_wm and not j_is_wm:
                # Swap: use j - i instead of i - j
                anchor = j
                reference = i
                swap_needed = True
            else:
                # Normal case: use i - j
                anchor = i
                reference = j
                swap_needed = False
            
            # Load LFP data for both channels
            lfp_i, fs_i = _load_lfp_from_mat(available_channels[i])
            lfp_j, fs_j = _load_lfp_from_mat(available_channels[j])
            
            if lfp_i is None or lfp_j is None:
                continue
            
            # Ensure same length (should be same, but just in case)
            T = min(len(lfp_i), len(lfp_j))
            if T == 0:
                continue
            
            # Compute bipolar-referenced signal
            if swap_needed:
                # j - i (non-white matter minus white matter)
                lfp_bipolar = (lfp_j[:T] - lfp_i[:T]).astype(np.float64)
                fs = fs_j if fs_j is not None else fs_i
            else:
                # i - j (normal case)
                lfp_bipolar = (lfp_i[:T] - lfp_j[:T]).astype(np.float64)
                fs = fs_i if fs_i is not None else fs_j
            
            # Create output directory for this bipolar channel
            bipolar_name = f"b{bipolar_counter:03d}"
            bipolar_dir = output_dir / bipolar_name
            bipolar_dir.mkdir(exist_ok=True)
            
            # Save bipolar-referenced LFP
            # Always save as [anchor, reference] to show which was subtracted from which
            pair_array = np.array([anchor, reference], dtype=np.int32)
            if swap_needed:
                note = f"bipolar {pair_type}: ch{anchor:03d} - ch{reference:03d} (non-WM - WM, within bank)"
            else:
                note = f"bipolar {pair_type}: ch{anchor:03d} - ch{reference:03d} (within bank)"
            
            save_dict = {
                'lfp': lfp_bipolar,
                'pair': pair_array,
                'note': note
            }
            if fs is not None:
                save_dict['fs'] = float(fs)
            
            output_file = bipolar_dir / f"lfp_{bipolar_name}.mat"
            sio.savemat(str(output_file), save_dict, do_compression=True)
            
            # Also save a text file with pair information for easy reference
            pair_info_file = bipolar_dir / "pair_info.txt"
            with open(pair_info_file, 'w') as f:
                f.write(f"Bipolar channel: {bipolar_name}\n")
                f.write(f"Original channels: {anchor} - {reference}\n")
                f.write(f"Type: {pair_type}\n")
                f.write(f"Note: {note}\n")
                if swap_needed:
                    f.write(f"Note: Swapped to use non-white matter channel as anchor\n")
            
            # CRITICAL: Mark BOTH electrodes as used (they cannot be reused)
            # Mark both i and j as used, regardless of which was anchor
            used_electrodes.add(i)
            used_electrodes.add(j)
            pairs_used.append([anchor, reference])
            bipolar_counter += 1
    
    # Save summary
    pairs_used_array = np.asarray(pairs_used, dtype=int) if pairs_used else np.empty((0, 2), dtype=int)
    summary_file = output_dir / "pairs_used.mat"
    sio.savemat(str(summary_file), {'pairs_used': pairs_used_array}, do_compression=True)
    
    # Count horizontal vs vertical pairs
    n_horizontal = 0
    n_vertical = 0
    pairs_h_set = {tuple(p) for p in pairs_h}
    pairs_v_set = {tuple(p) for p in pairs_v}
    for pair in pairs_used:
        if tuple(pair) in pairs_h_set:
            n_horizontal += 1
        elif tuple(pair) in pairs_v_set:
            n_vertical += 1
    
    return {
        'output_dir': output_dir,
        'n_pairs_processed': len(pairs_used),
        'n_horizontal': n_horizontal,
        'n_vertical': n_vertical,
        'pairs_used': pairs_used_array
    }

def detect_ripples_on_bipolar_channels(
    trial_bipolar_dir: str | Path,
    *,
    fs: int = 1000,
    rp_band: Tuple[int, int] = (100, 140),
    order: int = 550,
    window_ms: int = 20,
    z_low: float = 2.5,
    z_outlier: float = 9.0,
    min_dur_ms: int = 30,
    merge_dur_ms: int = 10,
    epoch_ms: int = 200,
) -> Dict:
    """
    Detect ripples on all bipolar-referenced channels.
    
    For each bipolar channel folder (b001, b002, etc.):
    1. Load lfp_b###.mat
    2. Run detect_ripples() with specified parameters
    3. Save all DetectionResult fields to ripples_b###.mat
    
    Args:
        trial_bipolar_dir: Path to trial_bipolar directory (e.g., .../trial001_bipolar)
        fs: Sampling frequency (Hz), default 1000
        rp_band: Ripple frequency band (low, high) in Hz
        order: FIR filter order
        window_ms: RMS window size in milliseconds
        z_low: Z-score threshold for ripple detection
        z_outlier: Z-score threshold for outlier clipping
        min_dur_ms: Minimum ripple duration in milliseconds
        merge_dur_ms: Merge ripples closer than this (ms)
        epoch_ms: Epoch window size around peaks (ms)
        
    Returns:
        Dictionary with summary information
    """
    bipolar_path = Path(trial_bipolar_dir)
    if not bipolar_path.is_dir():
        raise ValueError(f"Bipolar directory not found: {bipolar_path}")
    
    # Find all bipolar channel folders
    bipolar_folders = sorted(bipolar_path.glob("b???"))
    if not bipolar_folders:
        raise ValueError(f"No bipolar channel folders found in {bipolar_path}")
    
    print(f"Found {len(bipolar_folders)} bipolar channels to process")
    
    # Statistics
    n_processed = 0
    n_failed = 0
    total_ripples = 0
    channels_with_ripples = 0
    channels_without_ripples = 0
    channel_summaries = []
    
    # Import here to avoid circular imports
    from ripple_core.analyze import detect_ripples
    
    # Process each bipolar channel
    for bipolar_folder in bipolar_folders:
        bipolar_name = bipolar_folder.name
        
        # Find lfp file
        lfp_file = bipolar_folder / f"lfp_{bipolar_name}.mat"
        if not lfp_file.exists():
            print(f"  Warning: {lfp_file} not found, skipping {bipolar_name}")
            n_failed += 1
            continue
        
        try:
            # Load LFP data
            data = sio.loadmat(str(lfp_file))
            lfp = data['lfp']
            
            # Ensure 1D
            if lfp.ndim > 1:
                lfp = lfp.flatten()
            lfp = lfp.astype(np.float64)
            
            # Get sampling rate (use provided fs if not in file)
            file_fs = fs
            if 'fs' in data:
                fs_val = data['fs']
                if isinstance(fs_val, np.ndarray):
                    file_fs = float(fs_val.item())
                else:
                    file_fs = float(fs_val)
                if np.isnan(file_fs) or file_fs <= 0:
                    file_fs = fs
            
            # Get pair information
            pair_info = data.get('pair', np.array([0, 0]))
            if isinstance(pair_info, np.ndarray) and pair_info.size >= 2:
                # Handle both 1D and 2D arrays
                pair_flat = pair_info.flatten()
                pair_anchor, pair_ref = int(pair_flat[0]), int(pair_flat[1])
            else:
                pair_anchor, pair_ref = 0, 0
            
            # Detect ripples
            result = detect_ripples(
                lfp,
                fs=file_fs,
                rp_band=rp_band,
                order=order,
                window_ms=window_ms,
                z_low=z_low,
                z_outlier=z_outlier,
                min_dur_ms=min_dur_ms,
                merge_dur_ms=merge_dur_ms,
                epoch_ms=epoch_ms,
            )
            
            n_ripples = len(result.peak_idx)
            total_ripples += n_ripples
            
            if n_ripples > 0:
                channels_with_ripples += 1
            else:
                channels_without_ripples += 1
            
            # Save ripple detection results with z_low in filename
            ripple_file = bipolar_folder / f"ripples_{bipolar_name}_zlow{z_low:.1f}.mat"
            save_dict = {
                # DetectionResult fields
                'bp_lfp': result.bp_lfp,
                'env_rip': result.env_rip,
                'bits': result.bits,
                'mu': result.mu,
                'sd': result.sd,
                'mu_og': result.mu_og,
                'sd_og': result.sd_og,
                'peak_idx': result.peak_idx,
                'real_duration': result.real_duration,
                'windowed_duration': result.windowed_duration,
                'raw_windowed_lfp': result.raw_windowed_lfp,
                'bp_windowed_lfp': result.bp_windowed_lfp,
                'merged_starts': result.merged_starts,
                'merged_ends': result.merged_ends,
                # Metadata
                'fs': float(file_fs),
                'pair': np.array([pair_anchor, pair_ref], dtype=np.int32),
                'n_ripples': n_ripples,
                # Detection parameters
                'rp_band': np.array(rp_band, dtype=np.int32),
                'order': order,
                'window_ms': window_ms,
                'z_low': z_low,
                'z_outlier': z_outlier,
                'min_dur_ms': min_dur_ms,
                'merge_dur_ms': merge_dur_ms,
                'epoch_ms': epoch_ms,
            }
            
            sio.savemat(str(ripple_file), save_dict, do_compression=True)
            
            # Store channel summary
            channel_summaries.append({
                'channel': bipolar_name,
                'pair': [pair_anchor, pair_ref],
                'n_ripples': n_ripples,
                'fs': file_fs,
                'lfp_length': len(lfp),
            })
            
            n_processed += 1
            
            if (n_processed + n_failed) % 10 == 0:
                print(f"  Processed {n_processed + n_failed}/{len(bipolar_folders)} channels...")
        
        except Exception as e:
            print(f"  Error processing {bipolar_name}: {e}")
            import traceback
            traceback.print_exc()
            n_failed += 1
            continue
    
    # Calculate statistics
    avg_ripples = total_ripples / n_processed if n_processed > 0 else 0.0
    
    # Save summary file with z_low in filename
    summary_file = bipolar_path / f"ripple_detection_summary_zlow{z_low:.1f}.mat"
    summary_data = {
        'n_channels_processed': n_processed,
        'n_channels_failed': n_failed,
        'total_ripples': total_ripples,
        'avg_ripples_per_channel': avg_ripples,
        'channels_with_ripples': channels_with_ripples,
        'channels_without_ripples': channels_without_ripples,
        'detection_params_fs': fs,
        'detection_params_rp_band': np.array(rp_band, dtype=np.int32),
        'detection_params_order': order,
        'detection_params_window_ms': window_ms,
        'detection_params_z_low': z_low,
        'detection_params_z_outlier': z_outlier,
        'detection_params_min_dur_ms': min_dur_ms,
        'detection_params_merge_dur_ms': merge_dur_ms,
        'detection_params_epoch_ms': epoch_ms,
    }
    
    # Add per-channel summary as structured array
    if channel_summaries:
        # Create structured array for MATLAB compatibility
        dtype = [
            ('channel', 'U10'),
            ('pair_anchor', int),
            ('pair_ref', int),
            ('n_ripples', int),
            ('fs', float),
            ('lfp_length', int),
        ]
        channel_array = np.array(
            [
                (
                    cs['channel'],
                    cs['pair'][0],
                    cs['pair'][1],
                    cs['n_ripples'],
                    cs['fs'],
                    cs['lfp_length'],
                )
                for cs in channel_summaries
            ],
            dtype=dtype
        )
        summary_data['channel_summaries'] = channel_array
    
    sio.savemat(str(summary_file), summary_data, do_compression=True)
    
    print("\n" + "=" * 70)
    print("Ripple Detection Summary")
    print("=" * 70)
    print(f"Channels processed: {n_processed}")
    print(f"Channels failed: {n_failed}")
    print(f"Total ripples detected: {total_ripples}")
    print(f"Average ripples per channel: {avg_ripples:.2f}")
    print(f"Channels with ripples: {channels_with_ripples}")
    print(f"Channels without ripples: {channels_without_ripples}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)
    
    return {
        'n_channels_processed': n_processed,
        'n_channels_failed': n_failed,
        'total_ripples': total_ripples,
        'avg_ripples_per_channel': avg_ripples,
        'channels_with_ripples': channels_with_ripples,
        'channels_without_ripples': channels_without_ripples,
        'summary_file': summary_file
    }
