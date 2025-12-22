# ripple_core/signal/process_bipolar.py
"""
Bipolar referencing processing pipeline for neural data analysis.

This module provides functions to:
1. Apply bipolar referencing to LFP data using spatial electrode layout
2. Process bipolar referenced data for ripple analysis
3. Load and manage bipolar referenced datasets
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.io as sio
from dataclasses import dataclass

from .pairs import make_bipolar_pairs_from_grid
from .reref import reref_trial


@dataclass
class BipolarChannel:
    """Container for bipolar referenced channel information."""
    channel_id: int
    pair: Tuple[int, int]  # (anchor, reference) channels
    lfp_data: np.ndarray
    fs: float
    note: str
    is_horizontal: bool = True


@dataclass
class BipolarDataset:
    """Container for bipolar referenced dataset."""
    channels: Dict[int, BipolarChannel]
    pairs_used: np.ndarray
    output_dir: Path
    n_horizontal: int
    n_vertical: int
    n_total: int


def process_bipolar_referencing(
    root_dir: Union[str, Path],
    bad_channels: Optional[List[int]] = None,
    prefer: str = "horizontal",
    output_suffix: str = "_re-referenced"
) -> BipolarDataset:
    """
    Process bipolar referencing for a trial directory.
    
    Args:
        root_dir: Path to trial directory containing chan###/lfp.mat files
        bad_channels: List of bad channel IDs to exclude
        prefer: "horizontal" or "vertical" - which pairs to process first
        output_suffix: Suffix for output directory
        
    Returns:
        BipolarDataset containing all processed bipolar channels
    """
    if bad_channels is None:
        bad_channels = []
    
    # Apply bipolar referencing
    result = reref_trial(root_dir, bad_channels, prefer)
    
    # Load the processed data
    return load_bipolar_dataset(result["out_dir"])


def load_bipolar_dataset(output_dir: Union[str, Path]) -> BipolarDataset:
    """
    Load a previously processed bipolar referenced dataset.
    
    Args:
        output_dir: Path to the _re-referenced directory
        
    Returns:
        BipolarDataset containing all bipolar channels
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Bipolar dataset not found: {output_dir}")
    
    # Load pairs_used.mat
    pairs_file = output_dir / "pairs_used.mat"
    if pairs_file.exists():
        pairs_data = sio.loadmat(pairs_file, squeeze_me=True)
        pairs_used = pairs_data["pairs_used"]
        if pairs_used.ndim == 1:
            pairs_used = pairs_used.reshape(1, -1)
    else:
        pairs_used = np.array([])
    
    # Load all bipolar channels
    channels = {}
    n_horizontal = 0
    n_vertical = 0
    
    for ch_dir in output_dir.glob("chan???_ref"):
        try:
            ch_id = int(ch_dir.name[4:7])
            
            # Load original bipolar channel data
            lfp_file = ch_dir / "lfp_ref.mat"
            if lfp_file.exists():
                data = sio.loadmat(lfp_file, squeeze_me=True)
                lfp_data = data["lfp_ref"].astype(np.float32)
                fs = float(data.get("fs", np.nan))
                pair = tuple(data["pair"].astype(int))
                note = str(data.get("note", ""))
                
                # Determine if horizontal or vertical
                is_horizontal = "horizontal" in note.lower()
                if is_horizontal:
                    n_horizontal += 1
                else:
                    n_vertical += 1
                
                channels[ch_id] = BipolarChannel(
                    channel_id=ch_id,
                    pair=pair,
                    lfp_data=lfp_data,
                    fs=fs,
                    note=note,
                    is_horizontal=is_horizontal
                )
            
            # Load flipped bipolar channel data
            lfp_flipped_file = ch_dir / "lfp_ref_flipped.mat"
            if lfp_flipped_file.exists():
                data = sio.loadmat(lfp_flipped_file, squeeze_me=True)
                lfp_data = data["lfp_ref"].astype(np.float32)
                fs = float(data.get("fs", np.nan))
                pair = tuple(data["pair"].astype(int))
                note = str(data.get("note", ""))
                
                # Create a new channel ID for flipped version (add 1000 to distinguish)
                flipped_ch_id = ch_id + 1000
                
                # Determine if horizontal or vertical
                is_horizontal = "horizontal" in note.lower()
                if is_horizontal:
                    n_horizontal += 1
                else:
                    n_vertical += 1
                
                channels[flipped_ch_id] = BipolarChannel(
                    channel_id=flipped_ch_id,
                    pair=pair,
                    lfp_data=lfp_data,
                    fs=fs,
                    note=note,
                    is_horizontal=is_horizontal
                )
            
        except Exception as e:
            print(f"Warning: Could not load channel {ch_dir.name}: {e}")
            continue
    
    return BipolarDataset(
        channels=channels,
        pairs_used=pairs_used,
        output_dir=output_dir,
        n_horizontal=n_horizontal,
        n_vertical=n_vertical,
        n_total=len(channels)
    )


def get_bipolar_lfp_matrix(
    dataset: BipolarDataset,
    channels: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[int], float]:
    """
    Extract LFP matrix from bipolar dataset.
    
    Args:
        dataset: BipolarDataset to extract from
        channels: Specific channel IDs to extract (None for all)
        
    Returns:
        Tuple of (lfp_matrix, channel_ids, fs)
        lfp_matrix: (n_channels, n_samples) array
        channel_ids: List of channel IDs in order
        fs: Sampling frequency
    """
    if channels is None:
        channels = list(dataset.channels.keys())
    
    # Validate channels exist
    valid_channels = [ch for ch in channels if ch in dataset.channels]
    if not valid_channels:
        raise ValueError("No valid channels found")
    
    # Get data dimensions
    first_ch = dataset.channels[valid_channels[0]]
    n_samples = len(first_ch.lfp_data)
    fs = first_ch.fs
    
    # Build matrix
    lfp_matrix = np.zeros((len(valid_channels), n_samples), dtype=np.float32)
    
    for i, ch_id in enumerate(valid_channels):
        ch_data = dataset.channels[ch_id]
        if len(ch_data.lfp_data) != n_samples:
            # Handle different lengths by truncating
            min_len = min(len(ch_data.lfp_data), n_samples)
            lfp_matrix[i, :min_len] = ch_data.lfp_data[:min_len]
        else:
            lfp_matrix[i] = ch_data.lfp_data
    
    return lfp_matrix, valid_channels, fs


def save_bipolar_analysis(
    dataset: BipolarDataset,
    analysis_results: Dict,
    output_file: Union[str, Path]
) -> None:
    """
    Save bipolar analysis results.
    
    Args:
        dataset: BipolarDataset used for analysis
        analysis_results: Dictionary containing analysis results
        output_file: Path to save results
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        "bipolar_dataset_info": {
            "n_horizontal": dataset.n_horizontal,
            "n_vertical": dataset.n_vertical,
            "n_total": dataset.n_total,
            "output_dir": str(dataset.output_dir),
            "pairs_used": dataset.pairs_used
        },
        **analysis_results
    }
    
    sio.savemat(output_file, save_data, do_compression=True)
    print(f"Bipolar analysis results saved to: {output_file}")


def create_bipolar_processing_script(
    trial_dir: Union[str, Path],
    output_dir: Union[str, Path],
    bad_channels: Optional[List[int]] = None
) -> None:
    """
    Create a complete bipolar processing script for a trial.
    
    Args:
        trial_dir: Path to trial directory with chan###/lfp.mat files
        output_dir: Path to save processed results
        bad_channels: List of bad channels to exclude
    """
    trial_dir = Path(trial_dir)
    output_dir = Path(output_dir)
    
    print(f"Processing bipolar referencing for: {trial_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process bipolar referencing
    dataset = process_bipolar_referencing(
        trial_dir, 
        bad_channels=bad_channels,
        prefer="horizontal"
    )
    
    print(f"Processed {dataset.n_total} bipolar channels:")
    print(f"  - Horizontal pairs: {dataset.n_horizontal}")
    print(f"  - Vertical pairs: {dataset.n_vertical}")
    
    # Extract LFP matrix for further analysis
    lfp_matrix, channel_ids, fs = get_bipolar_lfp_matrix(dataset)
    
    print(f"LFP matrix shape: {lfp_matrix.shape}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {lfp_matrix.shape[1] / fs:.2f} seconds")
    
    # Save processing summary
    summary = {
        "lfp_matrix": lfp_matrix,
        "channel_ids": np.array(channel_ids),
        "fs": fs,
        "n_samples": lfp_matrix.shape[1],
        "duration_sec": lfp_matrix.shape[1] / fs
    }
    
    save_bipolar_analysis(dataset, summary, output_dir / "bipolar_analysis.mat")
    
    print(f"Bipolar processing complete. Results saved to: {output_dir}")


# Example usage and testing functions
def test_bipolar_processing():
    """Test function for bipolar processing pipeline."""
    # This would be used with actual data
    print("Bipolar processing pipeline ready for testing with real data.")
    print("Use create_bipolar_processing_script() to process your trial data.")


if __name__ == "__main__":
    test_bipolar_processing()
