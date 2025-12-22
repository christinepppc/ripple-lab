"""
Laplacian re-referencing for neural data.

This module implements Laplacian re-referencing where each channel is referenced
to the average of its immediate spatial neighbors, excluding white matter channels.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set
from dataclasses import dataclass

# White matter channels to exclude from Laplacian referencing
WHITE_MATTER_CHANNELS = {
    2, 10, 14, 16, 25, 26, 28, 36, 39, 42, 46, 53, 54, 55, 57, 59, 60, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 85, 87, 94, 95, 100, 101, 102, 106, 111, 117, 125, 128, 131, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 163, 164, 165, 167, 168, 171, 173, 174, 175, 176, 178, 180, 182, 184, 192, 198, 199, 200, 201, 202, 204, 207, 208, 210, 212, 215, 216, 218
}

# Exact 26 x 10 grid from your MATLAB (same as bipolar)
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


@dataclass
class LaplacianChannel:
    """Information about a Laplacian-referenced channel."""
    channel_id: int
    neighbors: List[int]
    lfp_laplacian: np.ndarray
    fs: float
    n_samples: int
    duration_sec: float


@dataclass
class LaplacianDataset:
    """Dataset containing Laplacian-referenced channels."""
    channels: Dict[int, LaplacianChannel]
    fs: float
    n_samples: int
    duration_sec: float


def _bank_of(ch: int) -> int:
    """Determine which bank a channel belongs to."""
    if 1 <= ch <= 32:
        return 1
    elif 33 <= ch <= 64:
        return 2
    elif 65 <= ch <= 96:
        return 3
    elif 97 <= ch <= 128:
        return 4
    elif 129 <= ch <= 160:
        return 5
    elif 161 <= ch <= 192:
        return 6
    elif 193 <= ch <= 220:
        return 7
    else:
        return -1


def _find_single_mat(ch_dir: Path) -> Path | None:
    """Find exactly one .mat under chan### directory."""
    # First try to find lfp.mat (original expected format)
    cand = list(ch_dir.glob("lfp.mat"))
    if len(cand) == 1:
        return cand[0]
    
    # For the actual data structure, look for .mat files with session/trial info
    # Pattern: sess###_trial###_chan###_*.mat
    cand = list(ch_dir.glob("sess*_trial*_chan*.mat"))
    if len(cand) == 1:
        return cand[0]
    
    # Fallback: any single .mat file
    cand = list(ch_dir.glob("*.mat"))
    if len(cand) == 1:
        return cand[0]
    return None


def _load_lfp_from_mat(mat_path: Path) -> Tuple[np.ndarray, float]:
    """Load LFP data from .mat file, handling both v7.0 and v7.3 formats."""
    try:
        # Try scipy.io first (for v7.0 files)
        S = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception:
        # If that fails, try h5py for v7.3 files
        try:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                S = {}
                for key in f.keys():
                    if not key.startswith('__'):
                        S[key] = f[key][:]
        except ImportError:
            raise ImportError("Please install h5py to read MATLAB v7.3 files: pip install h5py")
        except Exception as e:
            raise IOError(f"Could not read {mat_path}: {e}")
    
    if "lfp" not in S:
        raise KeyError(f"'lfp' missing in {mat_path}")
    
    x = np.asarray(S["lfp"]).astype(np.float32).ravel()
    fs = S.get("fs", 1000.0)
    
    # Handle NaN sampling frequency
    if np.isnan(fs) or fs <= 0:
        fs = 1000.0
    
    return x, fs


def _find_spatial_neighbors_within_bank(channel_id: int, grid: np.ndarray, excluded_channels: Set[int]) -> List[int]:
    """
    Find spatial neighbors for a given channel within the same bank, excluding white matter channels.
    
    Args:
        channel_id: Channel ID to find neighbors for
        grid: 26x10 electrode grid
        excluded_channels: Set of channels to exclude (white matter + bad channels)
        
    Returns:
        List of neighbor channel IDs within the same bank
    """
    neighbors = []
    R, C = grid.shape
    
    # Find the position of the channel in the grid
    ch_row, ch_col = -1, -1
    for r in range(R):
        for c in range(C):
            if grid[r, c] == channel_id:
                ch_row, ch_col = r, c
                break
        if ch_row != -1:
            break
    
    if ch_row == -1:
        return neighbors  # Channel not found in grid
    
    # Get the bank of the current channel
    current_bank = _bank_of(channel_id)
    if current_bank == -1:
        return neighbors  # Channel not in any bank
    
    # Check all 8 surrounding positions (including diagonals)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip the channel itself
            
            new_row = ch_row + dr
            new_col = ch_col + dc
            
            # Check bounds
            if 0 <= new_row < R and 0 <= new_col < C:
                neighbor_id = grid[new_row, new_col]
                
                # Skip if neighbor is 0 (empty position) or excluded
                if neighbor_id != 0 and neighbor_id not in excluded_channels:
                    # Check if neighbor is in the same bank
                    neighbor_bank = _bank_of(neighbor_id)
                    if neighbor_bank == current_bank:
                        neighbors.append(neighbor_id)
    
    return neighbors


def process_laplacian_referencing(root_dir: str | Path, bad_channels: Iterable[int] = ()) -> Dict:
    """
    Process Laplacian re-referencing for all channels within each bank, excluding white matter channels.
    
    Args:
        root_dir: Root directory containing chan### subdirectories
        bad_channels: List of bad channels to exclude
        
    Returns:
        Dictionary with processing results
    """
    root_dir = Path(root_dir)
    bad_channels = set(bad_channels)
    excluded_channels = bad_channels | WHITE_MATTER_CHANNELS
    
    print("=== Laplacian Re-referencing Processing (Within-Bank) ===")
    print(f"Root directory: {root_dir}")
    print(f"Bad channels: {sorted(bad_channels)}")
    print(f"White matter channels excluded: {len(WHITE_MATTER_CHANNELS)} channels")
    print(f"Total excluded channels: {len(excluded_channels)}")
    print("Processing within each bank, excluding white matter channels")
    
    # Discover all channel directories
    chan_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("chan")])
    print(f"Found {len(chan_dirs)} channel directories")
    
    # Load LFP data for all available channels
    channel_data = {}
    channel_fs = {}
    bank_channels = {}
    
    for chan_dir in chan_dirs:
        try:
            # Extract channel number
            chan_num = int(chan_dir.name[4:])  # Remove "chan" prefix
            
            # Skip excluded channels
            if chan_num in excluded_channels:
                if chan_num in WHITE_MATTER_CHANNELS:
                    print(f"  Skipping white matter channel {chan_num}")
                else:
                    print(f"  Skipping bad channel {chan_num}")
                continue
                
            # Find .mat file
            mat_file = _find_single_mat(chan_dir)
            if mat_file is None:
                print(f"  Warning: No .mat file found in {chan_dir}")
                continue
            
            # Load LFP data
            lfp, fs = _load_lfp_from_mat(mat_file)
            channel_data[chan_num] = lfp
            channel_fs[chan_num] = fs
            
            # Group by bank
            bank = _bank_of(chan_num)
            if bank not in bank_channels:
                bank_channels[bank] = []
            bank_channels[bank].append(chan_num)
            
            print(f"  Channel {chan_num} loaded -> Bank {bank}")
            
        except Exception as e:
            print(f"  Error processing {chan_dir}: {e}")
            continue
    
    print(f"\nLoaded {len(channel_data)} channels for Laplacian processing")
    print(f"Channels grouped into {len(bank_channels)} banks:")
    for bank, channels in bank_channels.items():
        print(f"  Bank {bank}: {len(channels)} channels")
    
    # Process Laplacian referencing for each channel within its bank
    laplacian_results = {}
    total_channels_processed = 0
    channels_without_neighbors = 0
    bank_stats = {}
    
    for bank, channels in bank_channels.items():
        print(f"\n=== Processing Bank {bank} ({len(channels)} channels) ===")
        bank_processed = 0
        bank_no_neighbors = 0
        
        for channel_id in sorted(channels):
            print(f"\nProcessing channel {channel_id} (Bank {bank})...")
            
            # Find spatial neighbors within the same bank
            neighbors = _find_spatial_neighbors_within_bank(channel_id, _LAYOUT_26x10, excluded_channels)
            
            if not neighbors:
                print(f"  No valid neighbors found for channel {channel_id} in Bank {bank}")
                channels_without_neighbors += 1
                bank_no_neighbors += 1
                continue
            
            # Check if all neighbors have data
            valid_neighbors = []
            for neighbor_id in neighbors:
                if neighbor_id in channel_data:
                    valid_neighbors.append(neighbor_id)
                else:
                    print(f"  Warning: Neighbor {neighbor_id} not available")
            
            if not valid_neighbors:
                print(f"  No valid neighbors with data for channel {channel_id}")
                channels_without_neighbors += 1
                bank_no_neighbors += 1
                continue
            
            print(f"  Found {len(valid_neighbors)} valid neighbors in Bank {bank}: {valid_neighbors}")
            
            # Compute Laplacian: channel - mean(neighbors)
            lfp_data = channel_data[channel_id]
            neighbor_data = [channel_data[nid] for nid in valid_neighbors]
            min_length = min(len(lfp_data), *[len(nd) for nd in neighbor_data])
            
            if min_length == 0:
                print(f"  Warning: No valid data for channel {channel_id}")
                continue
            
            # Truncate all data to same length
            lfp_truncated = lfp_data[:min_length]
            neighbor_matrix = np.array([nd[:min_length] for nd in neighbor_data])
            
            # Compute Laplacian
            neighbor_mean = np.mean(neighbor_matrix, axis=0)
            lfp_laplacian = lfp_truncated - neighbor_mean
            
            # Store results
            fs = channel_fs[channel_id]
            laplacian_channel = LaplacianChannel(
                channel_id=channel_id,
                neighbors=valid_neighbors,
                lfp_laplacian=lfp_laplacian,
                fs=fs,
                n_samples=len(lfp_laplacian),
                duration_sec=len(lfp_laplacian) / fs
            )
            
            laplacian_results[channel_id] = laplacian_channel
            total_channels_processed += 1
            bank_processed += 1
            
            print(f"  ✓ Channel {channel_id}: {len(lfp_laplacian)} samples, {len(valid_neighbors)} neighbors")
        
        bank_stats[bank] = {
            'total_channels': len(channels),
            'processed': bank_processed,
            'no_neighbors': bank_no_neighbors
        }
        print(f"\nBank {bank} summary: {bank_processed}/{len(channels)} channels processed, {bank_no_neighbors} without neighbors")
    
    # Create output directory structure
    output_dir = root_dir.parent / (root_dir.name + "_laplacian_processed")
    output_dir.mkdir(exist_ok=True)
    
    # Save Laplacian data for each channel
    print(f"\nSaving Laplacian data to {output_dir}...")
    
    for channel_id, laplacian_channel in laplacian_results.items():
        # Create channel directory
        ch_dir = output_dir / f"chan{channel_id:03d}_laplacian"
        ch_dir.mkdir(exist_ok=True)
        
        # Save Laplacian LFP data
        sio.savemat(ch_dir / "lfp_laplacian.mat", {
            "lfp_laplacian": laplacian_channel.lfp_laplacian,
            "fs": laplacian_channel.fs,
            "channel_id": laplacian_channel.channel_id,
            "neighbors": np.array(laplacian_channel.neighbors, dtype=int),
            "n_neighbors": len(laplacian_channel.neighbors),
            "n_samples": laplacian_channel.n_samples,
            "duration_sec": laplacian_channel.duration_sec
        }, do_compression=True)
        
        print(f"  ✓ Channel {channel_id}: {laplacian_channel.n_samples} samples, {laplacian_channel.duration_sec:.2f}s")
    
    # Create consolidated LFP matrix
    create_laplacian_lfp_matrix(laplacian_results, output_dir)
    
    # Create summary
    summary = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "bad_channels": sorted(bad_channels),
        "white_matter_channels": sorted(WHITE_MATTER_CHANNELS),
        "total_channels_processed": total_channels_processed,
        "channels_without_neighbors": channels_without_neighbors,
        "channels_processed": sorted(laplacian_results.keys()),
        "bank_statistics": bank_stats,
        "processing_method": "within_bank_laplacian"
    }
    
    # Save summary
    sio.savemat(output_dir / "laplacian_processing_summary.mat", summary, do_compression=True)
    
    with open(output_dir / "laplacian_processing_summary.txt", "w") as f:
        f.write("Laplacian Re-referencing Processing Summary (Within-Bank)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Root directory: {root_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Bad channels: {sorted(bad_channels)}\n")
        f.write(f"White matter channels excluded: {len(WHITE_MATTER_CHANNELS)}\n")
        f.write(f"Total channels processed: {total_channels_processed}\n")
        f.write(f"Channels without neighbors: {channels_without_neighbors}\n")
        f.write(f"Processing method: Within-bank Laplacian (8-neighbor spatial)\n")
        f.write(f"\nBank Statistics:\n")
        for bank, stats in bank_stats.items():
            f.write(f"  Bank {bank}: {stats['processed']}/{stats['total_channels']} processed, {stats['no_neighbors']} without neighbors\n")
        f.write(f"\nChannels processed: {sorted(laplacian_results.keys())}\n")
    
    print(f"\n✓ Laplacian processing completed!")
    print(f"  - Total channels processed: {total_channels_processed}")
    print(f"  - Channels without neighbors: {channels_without_neighbors}")
    print(f"  - White matter channels excluded: {len(WHITE_MATTER_CHANNELS)}")
    print(f"  - Processing method: Within-bank Laplacian (8-neighbor spatial)")
    print(f"  - Results saved to: {output_dir}")
    
    return summary


def create_laplacian_lfp_matrix(laplacian_results: Dict, output_dir: Path) -> None:
    """
    Create a consolidated LFP matrix from Laplacian results and save it.
    
    Args:
        laplacian_results: Dictionary containing Laplacian results for each channel
        output_dir: Output directory to save the matrix
    """
    print("\nCreating consolidated Laplacian LFP matrix...")
    
    if not laplacian_results:
        print("  Warning: No channels found for matrix creation")
        return
    
    # Collect all channels and their data
    all_channels = []
    all_lfp_data = []
    all_fs = []
    
    for channel_id, laplacian_channel in laplacian_results.items():
        all_channels.append(channel_id)
        all_lfp_data.append(laplacian_channel.lfp_laplacian)
        all_fs.append(laplacian_channel.fs)
    
    # Convert to numpy arrays
    channel_ids = np.array(all_channels, dtype=int)
    lfp_matrix = np.array(all_lfp_data, dtype=np.float32)
    fs_values = np.array(all_fs)
    
    # Use the most common sampling frequency
    fs = float(np.median(fs_values))
    
    print(f"  Matrix shape: {lfp_matrix.shape}")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Duration: {lfp_matrix.shape[1] / fs:.2f} seconds")
    
    # Save consolidated matrix
    matrix_file = output_dir / "laplacian_lfp_matrix.mat"
    sio.savemat(matrix_file, {
        "lfp_matrix": lfp_matrix,
        "channel_ids": channel_ids,
        "fs": fs,
        "n_samples": lfp_matrix.shape[1],
        "duration_sec": lfp_matrix.shape[1] / fs,
        "white_matter_excluded": True,
        "n_white_matter_excluded": len(WHITE_MATTER_CHANNELS)
    }, do_compression=True)
    
    print(f"  ✓ Laplacian LFP matrix saved to: {matrix_file}")


def load_laplacian_dataset(output_dir: Path) -> LaplacianDataset:
    """
    Load a previously processed Laplacian dataset.
    
    Args:
        output_dir: Path to the _laplacian_processed directory
        
    Returns:
        LaplacianDataset containing all Laplacian channels
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"Laplacian dataset not found: {output_dir}")
    
    # Try to load consolidated matrix first
    matrix_file = output_dir / "laplacian_lfp_matrix.mat"
    if matrix_file.exists():
        data = sio.loadmat(matrix_file, squeeze_me=True)
        lfp_matrix = data["lfp_matrix"]
        channel_ids = data["channel_ids"].flatten()
        fs = float(data.get("fs", 1000.0))
        
        # Create LaplacianChannel objects
        channels = {}
        
        for i, ch_id in enumerate(channel_ids):
            channels[ch_id] = LaplacianChannel(
                channel_id=ch_id,
                neighbors=[],  # Will be loaded from individual files if needed
                lfp_laplacian=lfp_matrix[i],
                fs=fs,
                n_samples=len(lfp_matrix[i]),
                duration_sec=len(lfp_matrix[i]) / fs
            )
        
        return LaplacianDataset(
            channels=channels,
            fs=fs,
            n_samples=lfp_matrix.shape[1],
            duration_sec=lfp_matrix.shape[1] / fs
        )
    
    # Fallback: load from individual channel directories
    channels = {}
    all_fs = []
    
    for ch_dir in output_dir.glob("chan*_laplacian"):
        ch_id = int(ch_dir.name[4:13])  # Extract channel ID from "chan###_laplacian"
        lfp_file = ch_dir / "lfp_laplacian.mat"
        
        if lfp_file.exists():
            data = sio.loadmat(lfp_file, squeeze_me=True)
            lfp_laplacian = data["lfp_laplacian"]
            fs = float(data.get("fs", 1000.0))
            neighbors = data.get("neighbors", []).tolist() if data.get("neighbors") is not None else []
            all_fs.append(fs)
            
            channels[ch_id] = LaplacianChannel(
                channel_id=ch_id,
                neighbors=neighbors,
                lfp_laplacian=lfp_laplacian,
                fs=fs,
                n_samples=len(lfp_laplacian),
                duration_sec=len(lfp_laplacian) / fs
            )
    
    fs = float(np.median(all_fs)) if all_fs else 1000.0
    n_samples = max(len(ch.lfp_laplacian) for ch in channels.values()) if channels else 0
    
    return LaplacianDataset(
        channels=channels,
        fs=fs,
        n_samples=n_samples,
        duration_sec=n_samples / fs
    )


def get_laplacian_lfp_matrix(dataset: LaplacianDataset) -> Tuple[np.ndarray, List[int], float]:
    """
    Extract LFP matrix from Laplacian dataset.
    
    Args:
        dataset: LaplacianDataset object
        
    Returns:
        Tuple of (lfp_matrix, channel_ids, fs)
    """
    if not dataset.channels:
        raise ValueError("No valid channels found")
    
    # Get all channel IDs and sort them
    channel_ids = sorted(dataset.channels.keys())
    
    # Extract LFP data
    lfp_data = []
    for ch_id in channel_ids:
        lfp_data.append(dataset.channels[ch_id].lfp_laplacian)
    
    # Convert to matrix
    lfp_matrix = np.array(lfp_data)
    
    return lfp_matrix, channel_ids, dataset.fs


__all__ = [
    "LaplacianChannel",
    "LaplacianDataset",
    "process_laplacian_referencing",
    "load_laplacian_dataset",
    "get_laplacian_lfp_matrix",
    "WHITE_MATTER_CHANNELS"
]
