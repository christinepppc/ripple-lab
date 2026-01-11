#!/usr/bin/env python3
"""
Visualize ripple propagation to investigate noise vs real propagation.

Creates scatter plots showing:
1. All ripple detections on channel positions
2. Multi-channel events highlighted
3. Time delay analysis to identify potential noise
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_propagation.py <trial_bipolar_dir> <layout_file> [propagation_file]")
        print("\nExample:")
        print("  python visualize_propagation.py /path/to/trial001_bipolar /path/to/bipolar_layout.csv")
        sys.exit(1)
    
    trial_bipolar_dir = Path(sys.argv[1])
    layout_file = Path(sys.argv[2])
    propagation_file = trial_bipolar_dir / "propagation_analysis_zlow3.0.mat"
    
    if len(sys.argv) >= 4:
        propagation_file = Path(sys.argv[3])
    
    print("=" * 70)
    print("Ripple Propagation Visualization")
    print("=" * 70)
    
    # Load layout
    layout_df = pd.read_csv(layout_file)
    print(f"Loaded layout with {len(layout_df)} bipolar channels")
    
    # Load propagation results
    if not propagation_file.exists():
        print(f"ERROR: Propagation file not found: {propagation_file}")
        sys.exit(1)
    
    data = sio.loadmat(str(propagation_file), squeeze_me=True)
    print(f"Loaded propagation analysis with {int(data['n_events'])} events")
    
    # Extract data
    n_channels_per_event = data['n_channels_per_event']
    max_time_delays = data['max_time_delays']
    propagation_sequences = data['propagation_sequences']
    initiation_channels = data['initiation_channels']
    spatial_extents = data['spatial_extents']
    
    # Identify multi-channel events
    multi_channel_mask = n_channels_per_event > 1
    multi_channel_indices = np.where(multi_channel_mask)[0]
    
    # Separate events by time delay threshold (5ms)
    fast_events = multi_channel_indices[max_time_delays[multi_channel_indices] < 5.0]
    slow_events = multi_channel_indices[max_time_delays[multi_channel_indices] >= 5.0]
    
    print(f"\nMulti-channel events breakdown:")
    print(f"  Total multi-channel: {len(multi_channel_indices)}")
    print(f"  < 5ms delay (potential noise): {len(fast_events)}")
    print(f"  >= 5ms delay (potential propagation): {len(slow_events)}")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: All ripple detections colored by event type
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot all channel positions
    ax1.scatter(layout_df['X'], layout_df['Y'], 
                c='lightgray', s=50, alpha=0.3, marker='o', 
                edgecolors='black', linewidths=0.5, label='Channels')
    
    # Plot single-channel events (initiation channels)
    single_channel_mask = n_channels_per_event == 1
    single_init_channels = initiation_channels[single_channel_mask]
    single_channel_positions = []
    for ch in single_init_channels:
        ch_data = layout_df[layout_df['bipolar_ch'] == ch]
        if len(ch_data) > 0:
            single_channel_positions.append((ch_data.iloc[0]['X'], ch_data.iloc[0]['Y']))
    
    if single_channel_positions:
        single_x, single_y = zip(*single_channel_positions)
        ax1.scatter(single_x, single_y, c='blue', s=20, alpha=0.5, 
                   marker='o', label=f'Single-channel events ({np.sum(single_channel_mask)})')
    
    # Plot multi-channel events (initiation channels)
    multi_init_channels = initiation_channels[multi_channel_mask]
    multi_channel_positions = []
    for ch in multi_init_channels:
        ch_data = layout_df[layout_df['bipolar_ch'] == ch]
        if len(ch_data) > 0:
            multi_channel_positions.append((ch_data.iloc[0]['X'], ch_data.iloc[0]['Y']))
    
    if multi_channel_positions:
        multi_x, multi_y = zip(*multi_channel_positions)
        ax1.scatter(multi_x, multi_y, c='red', s=30, alpha=0.7, 
                   marker='s', label=f'Multi-channel events ({len(multi_channel_indices)})')
    
    ax1.set_xlabel('X (normalized)', fontsize=11)
    ax1.set_ylabel('Y (normalized)', fontsize=11)
    ax1.set_title('All Ripple Detections\n(Initiation Channels)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Multi-channel events colored by max time delay
    ax2 = plt.subplot(2, 3, 2)
    
    ax2.scatter(layout_df['X'], layout_df['Y'], 
                c='lightgray', s=50, alpha=0.3, marker='o', 
                edgecolors='black', linewidths=0.5)
    
    # Plot fast events (<5ms)
    fast_init_channels = initiation_channels[fast_events] if len(fast_events) > 0 else []
    fast_positions = []
    for ch in fast_init_channels:
        ch_data = layout_df[layout_df['bipolar_ch'] == ch]
        if len(ch_data) > 0:
            fast_positions.append((ch_data.iloc[0]['X'], ch_data.iloc[0]['Y']))
    
    if fast_positions:
        fast_x, fast_y = zip(*fast_positions)
        ax2.scatter(fast_x, fast_y, c='orange', s=50, alpha=0.7, 
                   marker='s', label=f'< 5ms delay ({len(fast_events)})')
    
    # Plot slow events (>=5ms)
    slow_init_channels = initiation_channels[slow_events] if len(slow_events) > 0 else []
    slow_positions = []
    for ch in slow_init_channels:
        ch_data = layout_df[layout_df['bipolar_ch'] == ch]
        if len(ch_data) > 0:
            slow_positions.append((ch_data.iloc[0]['X'], ch_data.iloc[0]['Y']))
    
    if slow_positions:
        slow_x, slow_y = zip(*slow_positions)
        ax2.scatter(slow_x, slow_y, c='purple', s=50, alpha=0.7, 
                   marker='s', label=f'>= 5ms delay ({len(slow_events)})')
    
    ax2.set_xlabel('X (normalized)', fontsize=11)
    ax2.set_ylabel('Y (normalized)', fontsize=11)
    ax2.set_title('Multi-Channel Events\n(Colored by Time Delay)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Plot 3: Histogram of max time delays
    ax3 = plt.subplot(2, 3, 3)
    
    multi_delays = max_time_delays[multi_channel_indices]
    ax3.hist(multi_delays, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='5ms threshold')
    ax3.set_xlabel('Max Time Delay (ms)', fontsize=11)
    ax3.set_ylabel('Number of Events', fontsize=11)
    ax3.set_title('Distribution of Max Time Delays\n(Multi-Channel Events)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Number of channels per event
    ax4 = plt.subplot(2, 3, 4)
    
    multi_n_channels = n_channels_per_event[multi_channel_indices]
    ax4.hist(multi_n_channels, bins=range(2, int(np.max(multi_n_channels)) + 2), 
             color='green', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Channels', fontsize=11)
    ax4.set_ylabel('Number of Events', fontsize=11)
    ax4.set_title('Channels per Multi-Channel Event', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Spatial extent vs time delay
    ax5 = plt.subplot(2, 3, 5)
    
    multi_spatial = spatial_extents[multi_channel_indices]
    multi_delays = max_time_delays[multi_channel_indices]
    
    # Color by delay threshold
    fast_mask = multi_delays < 5.0
    ax5.scatter(multi_spatial[fast_mask], multi_delays[fast_mask], 
               c='orange', s=50, alpha=0.6, label=f'< 5ms ({np.sum(fast_mask)})')
    ax5.scatter(multi_spatial[~fast_mask], multi_delays[~fast_mask], 
               c='purple', s=50, alpha=0.6, label=f'>= 5ms ({np.sum(~fast_mask)})')
    
    ax5.axhline(y=5.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_xlabel('Spatial Extent (normalized)', fontsize=11)
    ax5.set_ylabel('Max Time Delay (ms)', fontsize=11)
    ax5.set_title('Spatial Extent vs Time Delay', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Scatter of all channels with event density
    ax6 = plt.subplot(2, 3, 6)
    
    # Count events per channel
    all_init_channels = initiation_channels
    channel_event_counts = {}
    for ch in all_init_channels:
        channel_event_counts[ch] = channel_event_counts.get(ch, 0) + 1
    
    # Plot channels colored by event count
    for _, row in layout_df.iterrows():
        ch = row['bipolar_ch']
        count = channel_event_counts.get(ch, 0)
        if count > 0:
            ax6.scatter(row['X'], row['Y'], s=count*2, c='red', 
                       alpha=0.6, edgecolors='black', linewidths=0.5)
        else:
            ax6.scatter(row['X'], row['Y'], s=20, c='lightgray', 
                       alpha=0.3, edgecolors='black', linewidths=0.5)
    
    ax6.set_xlabel('X (normalized)', fontsize=11)
    ax6.set_ylabel('Y (normalized)', fontsize=11)
    ax6.set_title('Event Density per Channel\n(Size = number of events)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figure
    output_png = trial_bipolar_dir / "propagation_visualization.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_png}")
    
    # Print detailed statistics
    print("\n" + "=" * 70)
    print("Detailed Analysis")
    print("=" * 70)
    print(f"\nMulti-channel events with < 5ms delay:")
    print(f"  Count: {len(fast_events)}")
    if len(fast_events) > 0:
        fast_delays = max_time_delays[fast_events]
        fast_nch = n_channels_per_event[fast_events]
        fast_spatial = spatial_extents[fast_events]
        print(f"  Avg delay: {np.mean(fast_delays):.2f} ms")
        print(f"  Avg channels: {np.mean(fast_nch):.2f}")
        print(f"  Avg spatial extent: {np.mean(fast_spatial):.4f}")
    
    print(f"\nMulti-channel events with >= 5ms delay:")
    print(f"  Count: {len(slow_events)}")
    if len(slow_events) > 0:
        slow_delays = max_time_delays[slow_events]
        slow_nch = n_channels_per_event[slow_events]
        slow_spatial = spatial_extents[slow_events]
        print(f"  Avg delay: {np.mean(slow_delays):.2f} ms")
        print(f"  Avg channels: {np.mean(slow_nch):.2f}")
        print(f"  Avg spatial extent: {np.mean(slow_spatial):.4f}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
