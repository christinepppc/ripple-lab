"""
Common Average Reference (CAR) re-referencing for neural data.

This module implements CAR re-referencing where each channel is referenced
to the average of all other channels within the same bank, excluding white matter channels.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from dataclasses import dataclass

# White matter channels to exclude from CAR referencing
WHITE_MATTER_CHANNELS = {
    2, 10, 14, 16, 25, 26, 28, 36, 39, 42, 46, 53, 54, 55, 57, 59, 60, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 85, 87, 94, 95, 100, 101, 102, 106, 111, 117, 125, 128, 131, 134, 135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 163, 164, 165, 167, 168, 171, 173, 174, 175, 176, 178, 180, 182, 184, 192, 198, 199, 200, 201, 202, 204, 207, 208, 210, 212, 215, 216, 218
}


@dataclass
class CARChannel:
    """Information about a CAR-referenced channel."""
    channel_id: int
    bank: int
    lfp_car: np.ndarray
    fs: float
    n_samples: int
    duration_sec: float


@dataclass
class CARDataset:
    """Dataset containing CAR-referenced channels."""
    channels: Dict[int, CARChannel]
    bank_channels: Dict[int, List[int]]
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
    elif 193 <= ch <= 224:
        return 7
    else:
        return -1


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


def _find_single_mat(ch_dir: Path) -> Path | None:
    """Find exactly one .mat under chan### directory."""
    cand = list(ch_dir.glob("*.mat"))
    if len(cand) == 1:
        return cand[0]
    return None


def process_car_referencing(root_dir: str | Path, bad_channels: Iterable[int] = ()) -> Dict:
    """
    Process CAR re-referencing for all channels in each bank, excluding white matter channels.
    
    Args:
        root_dir: Root directory containing chan### subdirectories
        bad_channels: List of bad channels to exclude
        
    Returns:
        Dictionary with processing results
    """
    root_dir = Path(root_dir)
    bad_channels = set(bad_channels)
    
    print("=== CAR Re-referencing Processing ===")
    print(f"Root directory: {root_dir}")
    print(f"Bad channels: {sorted(bad_channels)}")
    print(f"White matter channels excluded: {len(WHITE_MATTER_CHANNELS)} channels")
    
    # Discover all channel directories
    chan_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("chan")])
    print(f"Found {len(chan_dirs)} channel directories")
    
    # Group channels by bank
    bank_channels = {}
    bank_lfp_data = {}
    bank_fs = {}
    
    for chan_dir in chan_dirs:
        try:
            # Extract channel number
            chan_num = int(chan_dir.name[4:])  # Remove "chan" prefix
            
            # Skip white matter channels
            if chan_num in WHITE_MATTER_CHANNELS:
                print(f"  Skipping white matter channel {chan_num}")
                continue
            
            # Skip bad channels
            if chan_num in bad_channels:
                print(f"  Skipping bad channel {chan_num}")
                continue
                
            # Find .mat file
            mat_file = _find_single_mat(chan_dir)
            if mat_file is None:
                print(f"  Warning: No .mat file found in {chan_dir}")
                continue
            
            # Load LFP data
            lfp, fs = _load_lfp_from_mat(mat_file)
            
            # Determine bank
            bank = _bank_of(chan_num)
            if bank == -1:
                print(f"  Warning: Channel {chan_num} not in any bank")
                continue
            
            # Store data
            if bank not in bank_channels:
                bank_channels[bank] = []
                bank_lfp_data[bank] = []
                bank_fs[bank] = fs
            
            bank_channels[bank].append(chan_num)
            bank_lfp_data[bank].append(lfp)
            
            print(f"  Channel {chan_num} -> Bank {bank}")
            
        except Exception as e:
            print(f"  Error processing {chan_dir}: {e}")
            continue
    
    print(f"\nChannels grouped into {len(bank_channels)} banks:")
    for bank, channels in bank_channels.items():
        print(f"  Bank {bank}: {len(channels)} channels")
    
    # Process CAR for each bank
    car_results = {}
    total_channels_processed = 0
    
    for bank, channels in bank_channels.items():
        print(f"\nProcessing Bank {bank} ({len(channels)} channels)...")
        
        # Get LFP data for this bank
        lfp_data = bank_lfp_data[bank]
        fs = bank_fs[bank]
        
        # Convert to numpy array
        lfp_matrix = np.array(lfp_data)  # Shape: (n_channels, n_samples)
        
        # Compute CAR: subtract mean of all channels from each channel
        car_lfp = lfp_matrix - np.mean(lfp_matrix, axis=0, keepdims=True)
        
        # Store results for this bank
        bank_results = {}
        for i, ch_id in enumerate(channels):
            car_channel = CARChannel(
                channel_id=ch_id,
                bank=bank,
                lfp_car=car_lfp[i],
                fs=fs,
                n_samples=len(car_lfp[i]),
                duration_sec=len(car_lfp[i]) / fs
            )
            bank_results[ch_id] = car_channel
            total_channels_processed += 1
        
        car_results[bank] = bank_results
        print(f"  ✓ Bank {bank}: {len(channels)} channels CAR-processed")
    
    # Create output directory structure
    output_dir = root_dir.parent / (root_dir.name + "_car_processed")
    output_dir.mkdir(exist_ok=True)
    
    # Save CAR data for each bank
    for bank, bank_results in car_results.items():
        bank_dir = output_dir / f"bank_{bank:02d}"
        bank_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving Bank {bank} data to {bank_dir}...")
        
        for ch_id, car_channel in bank_results.items():
            # Create channel directory
            ch_dir = bank_dir / f"chan{ch_id:03d}_car"
            ch_dir.mkdir(exist_ok=True)
            
            # Save CAR LFP data
            sio.savemat(ch_dir / "lfp_car.mat", {
                "lfp_car": car_channel.lfp_car,
                "fs": car_channel.fs,
                "channel_id": car_channel.channel_id,
                "bank": car_channel.bank,
                "n_samples": car_channel.n_samples,
                "duration_sec": car_channel.duration_sec
            }, do_compression=True)
            
            print(f"  ✓ Channel {ch_id}: {car_channel.n_samples} samples, {car_channel.duration_sec:.2f}s")
    
    # Create summary
    summary = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "bad_channels": sorted(bad_channels),
        "total_channels_processed": total_channels_processed,
        "banks_processed": len(car_results),
        "bank_summary": {
            bank: {
                "n_channels": len(channels),
                "channels": sorted(channels)
            }
            for bank, channels in bank_channels.items()
        }
    }
    
    # Create consolidated LFP matrix
    create_car_lfp_matrix(car_results, output_dir)
    
    # Save summary
    sio.savemat(output_dir / "car_processing_summary.mat", summary, do_compression=True)
    
    with open(output_dir / "car_processing_summary.txt", "w") as f:
        f.write("CAR Re-referencing Processing Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Root directory: {root_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Bad channels: {sorted(bad_channels)}\n")
        f.write(f"White matter channels excluded: {len(WHITE_MATTER_CHANNELS)}\n")
        f.write(f"Total channels processed: {total_channels_processed}\n")
        f.write(f"Banks processed: {len(car_results)}\n\n")
        
        for bank, channels in bank_channels.items():
            f.write(f"Bank {bank}: {len(channels)} channels\n")
            f.write(f"  Channels: {sorted(channels)}\n")
    
    print(f"\n✓ CAR processing completed!")
    print(f"  - Total channels processed: {total_channels_processed}")
    print(f"  - Banks processed: {len(car_results)}")
    print(f"  - White matter channels excluded: {len(WHITE_MATTER_CHANNELS)}")
    print(f"  - Results saved to: {output_dir}")
    
    return summary


def create_car_lfp_matrix(car_results: Dict, output_dir: Path) -> None:
    """
    Create a consolidated LFP matrix from CAR results and save it.
    
    Args:
        car_results: Dictionary containing CAR results for each bank
        output_dir: Output directory to save the matrix
    """
    print("\nCreating consolidated CAR LFP matrix...")
    
    # Collect all channels and their data
    all_channels = []
    all_lfp_data = []
    all_fs = []
    
    for bank, bank_results in car_results.items():
        for ch_id, car_channel in bank_results.items():
            all_channels.append(ch_id)
            all_lfp_data.append(car_channel.lfp_car)
            all_fs.append(car_channel.fs)
    
    if not all_channels:
        print("  Warning: No channels found for matrix creation")
        return
    
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
    matrix_file = output_dir / "car_lfp_matrix.mat"
    sio.savemat(matrix_file, {
        "lfp_matrix": lfp_matrix,
        "channel_ids": channel_ids,
        "fs": fs,
        "n_samples": lfp_matrix.shape[1],
        "duration_sec": lfp_matrix.shape[1] / fs,
        "white_matter_excluded": True,
        "n_white_matter_excluded": len(WHITE_MATTER_CHANNELS)
    }, do_compression=True)
    
    print(f"  ✓ CAR LFP matrix saved to: {matrix_file}")


def load_car_dataset(output_dir: Path) -> CARDataset:
    """
    Load a previously processed CAR dataset.
    
    Args:
        output_dir: Path to the _car_processed directory
        
    Returns:
        CARDataset containing all CAR channels
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"CAR dataset not found: {output_dir}")
    
    # Try to load consolidated matrix first
    matrix_file = output_dir / "car_lfp_matrix.mat"
    if matrix_file.exists():
        data = sio.loadmat(matrix_file, squeeze_me=True)
        lfp_matrix = data["lfp_matrix"]
        channel_ids = data["channel_ids"].flatten()
        fs = float(data.get("fs", 1000.0))
        
        # Create CARChannel objects
        channels = {}
        bank_channels = {}
        
        for i, ch_id in enumerate(channel_ids):
            bank = _bank_of(ch_id)
            if bank not in bank_channels:
                bank_channels[bank] = []
            bank_channels[bank].append(ch_id)
            
            channels[ch_id] = CARChannel(
                channel_id=ch_id,
                bank=bank,
                lfp_car=lfp_matrix[i],
                fs=fs,
                n_samples=len(lfp_matrix[i]),
                duration_sec=len(lfp_matrix[i]) / fs
            )
        
        return CARDataset(
            channels=channels,
            bank_channels=bank_channels,
            fs=fs,
            n_samples=lfp_matrix.shape[1],
            duration_sec=lfp_matrix.shape[1] / fs
        )
    
    # Fallback: load from individual bank directories
    channels = {}
    bank_channels = {}
    all_fs = []
    
    for bank_dir in output_dir.glob("bank_*"):
        bank_num = int(bank_dir.name.split("_")[1])
        bank_channels[bank_num] = []
        
        for ch_dir in bank_dir.glob("chan*_car"):
            ch_id = int(ch_dir.name[4:7])
            lfp_file = ch_dir / "lfp_car.mat"
            
            if lfp_file.exists():
                data = sio.loadmat(lfp_file, squeeze_me=True)
                lfp_car = data["lfp_car"]
                fs = float(data.get("fs", 1000.0))
                all_fs.append(fs)
                
                channels[ch_id] = CARChannel(
                    channel_id=ch_id,
                    bank=bank_num,
                    lfp_car=lfp_car,
                    fs=fs,
                    n_samples=len(lfp_car),
                    duration_sec=len(lfp_car) / fs
                )
                bank_channels[bank_num].append(ch_id)
    
    fs = float(np.median(all_fs)) if all_fs else 1000.0
    n_samples = max(len(ch.lfp_car) for ch in channels.values()) if channels else 0
    
    return CARDataset(
        channels=channels,
        bank_channels=bank_channels,
        fs=fs,
        n_samples=n_samples,
        duration_sec=n_samples / fs
    )




def get_car_lfp_matrix(dataset: CARDataset) -> Tuple[np.ndarray, List[int], float]:
    """
    Extract LFP matrix from CAR dataset.
    
    Args:
        dataset: CARDataset object
        
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
        lfp_data.append(dataset.channels[ch_id].lfp_car)
    
    # Convert to matrix
    lfp_matrix = np.array(lfp_data)
    
    return lfp_matrix, channel_ids, dataset.fs


def save_car_analysis(results: Dict, output_dir: str | Path) -> None:
    """Save CAR analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results summary
    sio.savemat(output_dir / "car_analysis_summary.mat", results, do_compression=True)
    
    print(f"✓ CAR analysis results saved to: {output_dir}")


def create_car_processing_script() -> str:
    """Create a standalone script for CAR processing."""
    return '''#!/usr/bin/env python3
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
        
        print(f"\\n✓ CAR processing completed successfully!")
        print(f"Results saved to: {summary['output_dir']}")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


__all__ = [
    "CARChannel",
    "CARDataset", 
    "process_car_referencing",
    "load_car_dataset",
    "get_car_lfp_matrix",
    "save_car_analysis",
    "create_car_processing_script"
]


