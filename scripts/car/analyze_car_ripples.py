#!/usr/bin/env python3
"""
Analyze CAR-referenced data for ripples.

This script loads CAR-processed data and runs ripple detection on all channels.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from ripple_core.ripple_core.signal.car import load_car_dataset, get_car_lfp_matrix
from ripple_core.ripple_core.analyze import detect_ripples


def analyze_car_ripples(car_dir: str, output_dir: str, ripple_params: dict = None):
    """
    Analyze CAR-referenced data for ripples.
    
    Args:
        car_dir: Path to CAR-processed directory
        output_dir: Directory to save results
        ripple_params: Parameters for ripple detection
    """
    
    if ripple_params is None:
        ripple_params = {
            "rp_band": (100, 140),
            "order": 550,
            "window_ms": 20,
            "z_low": 2.5,
            "z_outlier": 9.0,
            "min_dur_ms": 30,
            "merge_dur_ms": 10,
            "epoch_ms": 200
        }
    
    print("=== CAR Ripple Analysis ===")
    print(f"CAR directory: {car_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ripple parameters: {ripple_params}")
    
    # Load CAR dataset
    print("\n1. Loading CAR dataset...")
    dataset = load_car_dataset(car_dir)
    
    print(f"✓ Loaded {len(dataset.channels)} CAR channels")
    print(f"✓ Banks: {len(dataset.bank_channels)}")
    print(f"✓ Sampling frequency: {dataset.fs} Hz")
    print(f"✓ Duration: {dataset.duration_sec:.2f} seconds")
    
    # Extract LFP matrix
    print(f"\n2. Extracting LFP matrix...")
    lfp_matrix, channel_ids, fs = get_car_lfp_matrix(dataset)
    print(f"✓ LFP matrix shape: {lfp_matrix.shape}")
    
    # Process each channel
    print(f"\n3. Running ripple detection on {len(channel_ids)} channels...")
    
    results = {}
    total_ripples = 0
    successful_channels = 0
    bank_summary = {}
    
    for i, ch_id in enumerate(channel_ids):
        car_channel = dataset.channels[ch_id]
        bank = car_channel.bank
        
        print(f"  Processing channel {ch_id} (Bank {bank}) ({i+1}/{len(channel_ids)})...")
        
        lfp_signal = lfp_matrix[i]
        
        try:
            # Detect ripples
            det_result = detect_ripples(
                lfp_signal,
                fs=fs,
                **ripple_params
            )
            
            n_ripples = len(det_result.peak_idx)
            total_ripples += n_ripples
            successful_channels += 1
            
            print(f"    ✓ {n_ripples} ripples detected")
            
            # Store results
            results[ch_id] = {
                "n_ripples": n_ripples,
                "peak_idx": det_result.peak_idx,
                "real_duration": det_result.real_duration,
                "mu": det_result.mu,
                "sd": det_result.sd,
                "bank": bank,
                "success": True
            }
            
            # Update bank summary
            if bank not in bank_summary:
                bank_summary[bank] = {"channels": 0, "ripples": 0, "successful": 0}
            bank_summary[bank]["channels"] += 1
            bank_summary[bank]["ripples"] += n_ripples
            bank_summary[bank]["successful"] += 1
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[ch_id] = {
                "n_ripples": 0,
                "bank": bank,
                "error": str(e),
                "success": False
            }
            
            # Update bank summary
            if bank not in bank_summary:
                bank_summary[bank] = {"channels": 0, "ripples": 0, "successful": 0}
            bank_summary[bank]["channels"] += 1
    
    print(f"\n✓ Processing complete!")
    print(f"  - Successful channels: {successful_channels}/{len(channel_ids)}")
    print(f"  - Total ripples detected: {total_ripples}")
    
    # Print bank summary
    print(f"\nBank Summary:")
    for bank in sorted(bank_summary.keys()):
        summary = bank_summary[bank]
        print(f"  Bank {bank}: {summary['successful']}/{summary['channels']} channels, {summary['ripples']} ripples")
    
    # Save results
    print(f"\n4. Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary
    summary = {
        "car_dir": car_dir,
        "n_channels_processed": len(channel_ids),
        "n_successful": successful_channels,
        "total_ripples": total_ripples,
        "fs": fs,
        "ripple_params": ripple_params,
        "bank_summary": bank_summary,
        "channel_summary": {
            ch_id: {
                "n_ripples": res["n_ripples"],
                "bank": res["bank"],
                "success": res["success"],
                "error": res.get("error", None)
            }
            for ch_id, res in results.items()
        }
    }
    
    # Save summary
    sio.savemat(output_path / "car_ripple_summary.mat", summary, do_compression=True)
    print(f"✓ Summary saved to: {output_path / 'car_ripple_summary.mat'}")
    
    # Save detailed results for channels with ripples
    channels_with_ripples = [ch_id for ch_id, res in results.items() if res["success"] and res["n_ripples"] > 0]
    
    if channels_with_ripples:
        print(f"✓ Saving detailed results for {len(channels_with_ripples)} channels with ripples...")
        
        for ch_id in channels_with_ripples:
            res = results[ch_id]
            
            # Save detection results
            ch_output_file = output_path / f"chan{ch_id:03d}_car_ripples.mat"
            sio.savemat(ch_output_file, {
                "channel_id": ch_id,
                "bank": res["bank"],
                "n_ripples": res["n_ripples"],
                "peak_idx": res["peak_idx"],
                "real_duration": res["real_duration"],
                "mu": res["mu"],
                "sd": res["sd"],
                "fs": fs
            }, do_compression=True)
    
    # Print final summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"CAR directory: {car_dir}")
    print(f"Results saved to: {output_path}")
    print(f"Channels processed: {len(channel_ids)}")
    print(f"Successful detections: {successful_channels}")
    print(f"Total ripples found: {total_ripples}")
    
    # Show top channels with most ripples
    channels_with_ripples = [(ch_id, res["n_ripples"], res["bank"]) for ch_id, res in results.items() if res["success"] and res["n_ripples"] > 0]
    channels_with_ripples.sort(key=lambda x: x[1], reverse=True)
    
    if channels_with_ripples:
        print(f"\nTop 10 channels with most ripples:")
        for i, (ch_id, n_ripples, bank) in enumerate(channels_with_ripples[:10]):
            print(f"  {i+1:2d}. Channel {ch_id:3d} (Bank {bank}): {n_ripples:3d} ripples")
    
    return results, summary


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze CAR data for ripples")
    parser.add_argument("car_dir", help="Path to CAR-processed directory")
    parser.add_argument("--output", "-o", help="Output directory (default: car_dir + '_ripples')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        car_path = Path(args.car_dir)
        output_dir = str(car_path.parent / (car_path.name + "_ripples"))
    
    # Ripple parameters
    ripple_params = {
        "rp_band": (100, 140),
        "order": 550,
        "window_ms": 20,
        "z_low": 2.5,
        "z_outlier": 9.0,
        "min_dur_ms": 30,
        "merge_dur_ms": 10,
        "epoch_ms": 200
    }
    
    try:
        results, summary = analyze_car_ripples(
            car_dir=args.car_dir,
            output_dir=output_dir,
            ripple_params=ripple_params
        )
        
        print(f"\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


