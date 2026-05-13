#!/usr/bin/env python3
"""
Plot raster of detected ripples for a session (pre vs post)
Similar to MATLAB inspectRippleTime_reordered
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Plot ripple raster for a session')
parser.add_argument('--session', type=int, required=True, help='Session number')
parser.add_argument('--pre', type=int, required=True, help='Pre-stimulation trial number')
parser.add_argument('--post', type=int, required=True, help='Post-stimulation trial number')
args = parser.parse_args()

# Paths
BASE_DIR = Path("/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen")
SESSION = args.session
PRE_TRIAL = args.pre
POST_TRIAL = args.post

# Load data
def load_trial_data(session, trial):
    """Load ripples and channel labels for a trial"""
    trial_dir = BASE_DIR / f"session{session:03d}" / f"trial{trial:03d}_bipolar"
    
    # Load passed ripples
    ripples_pkl = trial_dir / "derived" / "ripples_zlow3.0_passed.pkl"
    ripples_df = pd.read_pickle(ripples_pkl)
    
    # Load channel labels
    labels_csv = trial_dir / "bipolar_channel_labels.csv"
    labels_df = pd.read_csv(labels_csv)
    
    # Determine primary region for each bipolar channel
    # Use region_type to classify
    def get_primary_region(row):
        rt = row['region_type']
        if 'within_parietal' in rt or 'parietal' in rt:
            return 'parietal'
        elif 'within_prefrontal' in rt or 'prefrontal' in rt:
            return 'prefrontal'
        elif 'within_motor' in rt or 'motor' in rt:
            return 'motor'
        elif 'within_basal_ganglia' in rt or 'basal_ganglia' in rt:
            return 'basal_ganglia'
        elif 'within_thalamus' in rt or 'thalamus' in rt:
            return 'thalamus'
        elif 'within_mtl' in rt or 'mtl' in rt:
            return 'mtl'
        elif 'within_amygdala' in rt or 'amygdala' in rt:
            return 'amygdala'
        else:
            return 'other'
    
    labels_df['primary_region'] = labels_df.apply(get_primary_region, axis=1)
    
    # Merge to get region info for each ripple
    ripples_df = ripples_df.merge(
        labels_df[['bipolar_channel', 'primary_region']], 
        left_on='channel_id', 
        right_on='bipolar_channel',
        how='left'
    )
    
    return ripples_df, labels_df

# Load both trials
print(f"Loading Session {SESSION:03d} data...")
pre_ripples, pre_labels = load_trial_data(SESSION, PRE_TRIAL)
post_ripples, post_labels = load_trial_data(SESSION, POST_TRIAL)

print(f"Pre (trial {PRE_TRIAL:03d}): {len(pre_ripples)} ripples across {pre_ripples['channel_id'].nunique()} channels")
print(f"Post (trial {POST_TRIAL:03d}): {len(post_ripples)} ripples across {post_ripples['channel_id'].nunique()} channels")

# Get unique channels from labels (all bipolar channels)
all_channels = pre_labels['bipolar_channel'].tolist()
n_channels = len(all_channels)

# Sort channels by region for better visualization
channel_region_map = dict(zip(pre_labels['bipolar_channel'], pre_labels['primary_region']))
channels_sorted = sorted(all_channels, key=lambda ch: (
    channel_region_map.get(ch, 'zzz'),  # Region first
    ch  # Then channel name
))

# Create channel index mapping
channel_to_row = {ch: i for i, ch in enumerate(channels_sorted)}

# Region colors
region_colors = {
    'parietal': '#1f77b4',      # Blue
    'prefrontal': '#ff7f0e',    # Orange
    'motor': '#2ca02c',         # Green
    'basal_ganglia': '#d62728', # Red
    'thalamus': '#9467bd',      # Purple
    'mtl': '#8c564b',           # Brown
    'amygdala': '#e377c2',      # Pink
    'other': '#7f7f7f',         # Gray
}

# Create figure with two subplots (pre on top, post on bottom)
fig, axes = plt.subplots(2, 1, figsize=(14, 16), sharex=False)
fig.suptitle(f'Session {SESSION:03d} Ripple Raster Plot (z_low=3.0, passed rejection)', 
             fontsize=14, fontweight='bold')

tick_half = 0.4  # Half-height of raster tick

for idx, (trial_name, ripples_df, ax) in enumerate([
    (f'Pre-Stimulation (Trial {PRE_TRIAL:03d})', pre_ripples, axes[0]),
    (f'Post-Stimulation (Trial {POST_TRIAL:03d})', post_ripples, axes[1])
]):
    ax.set_title(trial_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Bipolar Channel (sorted by region)', fontsize=11)
    
    ax.set_ylim([0.5, n_channels + 0.5])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Get recording duration (max peak_time)
    if len(ripples_df) > 0:
        max_time = ripples_df['peak_time'].max()
        ax.set_xlim([0, max_time])
    
    # Plot ripples for each channel
    for ch in channels_sorted:
        row_y = channel_to_row[ch] + 1  # 1-indexed for plotting
        region = channel_region_map.get(ch, 'other')
        color = region_colors.get(region, '#7f7f7f')
        
        # Get ripples for this channel
        ch_ripples = ripples_df[ripples_df['channel_id'] == ch]
        
        if len(ch_ripples) == 0:
            continue
        
        # Use start_time for raster ticks
        start_times = ch_ripples['start_time'].values
        n_ripples = len(start_times)
        
        # Create raster lines (vertical ticks)
        X = np.column_stack([start_times, start_times, np.full(n_ripples, np.nan)]).ravel()
        Y = np.column_stack([
            np.full(n_ripples, row_y - tick_half),
            np.full(n_ripples, row_y + tick_half),
            np.full(n_ripples, np.nan)
        ]).ravel()
        
        ax.plot(X, Y, color=color, linewidth=0.8, alpha=0.7)
    
    # Y-axis: show every Nth channel to avoid overcrowding
    step = max(1, n_channels // 20)  # Show ~20 labels max
    ytick_positions = list(range(1, n_channels + 1, step))
    ytick_labels = [channels_sorted[i-1] for i in ytick_positions]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=7)

# Add legend
handles = []
labels_list = []
for region in ['parietal', 'prefrontal', 'motor', 'basal_ganglia', 'thalamus', 'mtl', 'amygdala', 'other']:
    if region in [channel_region_map[ch] for ch in all_channels]:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color=region_colors[region], linewidth=2))
        labels_list.append(region.replace('_', ' ').title())

fig.legend(handles, labels_list, loc='upper center', ncol=8, 
           bbox_to_anchor=(0.5, 0.995), frameon=True, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save figure
output_dir = BASE_DIR / "visualizations"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"session{SESSION:03d}_ripple_raster.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved raster plot to: {output_file}")

# Also save as PDF for publication quality
output_pdf = output_dir / f"session{SESSION:03d}_ripple_raster.pdf"
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved PDF to: {output_pdf}")

plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for trial_name, ripples_df in [
    (f'Pre (Trial {PRE_TRIAL:03d})', pre_ripples),
    (f'Post (Trial {POST_TRIAL:03d})', post_ripples)
]:
    print(f"\n{trial_name}:")
    print(f"  Total ripples: {len(ripples_df)}")
    print(f"  Channels with ripples: {ripples_df['channel_id'].nunique()}")
    
    # Ripples per region
    region_counts = ripples_df.groupby('primary_region').size().sort_values(ascending=False)
    print("  Ripples by region:")
    for region, count in region_counts.items():
        pct = 100 * count / len(ripples_df)
        print(f"    {region:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Mean ripple rate
    if len(ripples_df) > 0:
        duration = ripples_df['peak_time'].max()
        rate = len(ripples_df) / duration
        print(f"  Recording duration: {duration:.1f} sec")
        print(f"  Overall ripple rate: {rate:.3f} events/sec")

print("\n" + "="*70)
