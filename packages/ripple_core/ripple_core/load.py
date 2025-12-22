# load.py
import os
import numpy as np
import scipy.io
from scipy.io import loadmat
# from gui.windows import MainWindow  # Removed circular import
from dataclasses import dataclass
from typing import List, Optional

# --------------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------------
# Base directory for monkey data
MONKEYDIR = '/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd'
# Number of channels and sampling parameters
nCh = 220              # total channels per recording
Fs = 1000              # sampling frequency (Hz)
bytes_per_sample = 4   # bytes per float32 sample

# --------------------------------------------------------------------------------
# Session metadata loader for Arichie Movie Data
# --------------------------------------------------------------------------------
def load_movie_database():
    """
    Locate and parse the MATLAB script Movie_Database.m in MONKEYDIR.

    Parses lines like:
      Session{ind} = {'180531',{'018','019'},ind};
    into a list of dicts:
      [
        { 'date':'180531', 'trials':['018','019'], 'index':ind },
        ...
      ]

    Returns:
      List[Dict]: sorted by session index (ascending)
    """
    file_name = 'Movie_Database.m'
    file_path = os.path.join(MONKEYDIR, 'm', file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Could not find {file_name} in {MONKEYDIR}")
    
    # Initiate a list of dictionaries, so when indexing, can directly retrieve a dictionary

    sessions = []
    session_counter = 1  # start indexing sessions at 1

    def _find_matching_brace(s: str, start: int) -> int:
        """Find the index of the matching '}' for the '{' at position start."""
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
        raise ValueError(f"No matching closing brace found starting at {start}")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('Session{'):
                continue
            # strip MATLAB comments
            code_line = line.split('%', 1)[0]
            # isolate the right-hand side of '='
            try:
                _, rhs = code_line.split('=', 1)
            except ValueError:
                continue
            # remove outer braces and semicolon
            rhs = rhs.strip().lstrip('{').rstrip('};').strip()
            # parse date between first pair of quotes
            q1 = rhs.find("'")
            q2 = rhs.find("'", q1 + 1)
            date_str = rhs[q1 + 1:q2]
            # locate trial-list braces
            open_idx = rhs.find('{', q2)
            close_idx = _find_matching_brace(rhs, open_idx)
            trials_inner = rhs[open_idx + 1:close_idx]
            trials = [t.strip().strip("'") for t in trials_inner.split(',') if t.strip()]
            # record entry with 1-based index
            sessions.append({'date': date_str, 'trials': trials, 'session': session_counter})
            session_counter += 1

    return sessions

# --------------------------------------------------------------------------------
# Method 1: load a specific session, trial, and channel (IMPLEMENTED)
# --------------------------------------------------------------------------------
def load_electrodes(pick_sess, rec, ch):
    """
    Load LFP samples for a single channel in a given session and trial.

    Parameters:
      pick_sess (int): index of session in the movie database list (0-based)
      rec        (int or str): trial number or zero-padded string (e.g. 1 or '001')
      ch         (int): channel index (1-based)

    Returns:
      np.ndarray: 1D array of float32 samples for the requested channel
    """
    # Load session info
    sessions = load_movie_database()
    sess = sessions[pick_sess]
    sess_day = sess['date']

    # Normalize trial string
    rec_str = str(rec).zfill(3) if isinstance(rec, int) else rec

    # Validate channel index
    if ch < 1 or ch > nCh:
        raise ValueError(f"Invalid channel selected: {ch}")

    # Build path to the .dat file
    file_path = os.path.join(
        MONKEYDIR,
        sess_day,
        rec_str,
        f"rec{rec_str}.Frontal_1.lfp.dat"
    )

    # Read raw binary data for all channels
    raw = np.fromfile(file_path, dtype=np.float32)
    # Reshape to (n_samples, nCh)
    try:
        data = raw.reshape(-1, nCh)
    except ValueError:
        raise IOError(f"File size does not match expected channels: {file_path}")

    # Extract requested channel (convert 1-based to 0-based)
    ch_data = data[:, ch - 1]
    print(f"Loaded {ch_data.size} samples from channel {ch}, trial {rec_str}.")
    return ch_data

# --------------------------------------------------------------------------------
# Method 2: load region specific  channels interactively (NOT IMPLEMENTED)
# --------------------------------------------------------------------------------
def load_region_data(pick_sess: int, rec, region_label: str) -> tuple[np.ndarray, str]:
    """
    Load LFP from all channels in a given anatomical region for a specific session & trial.

    Parameters:
      pick_sess   (int)            : index into your movie-database list
      rec         (int or str)     : trial number (e.g. 1) or zero-padded (e.g. '001')
      region_label(str)            : one of your dropdown strings,
                                      e.g. "(r) anterior amygdalar area"

    Returns:
      ch_data     (np.ndarray)     : shape (n_chans, n_samples)
      sess_day    (str)            : session folder name (date)
    """
    # 1) Session info
    sessions = load_movie_database()
    sess     = sessions[pick_sess]
    sess_day = sess["date"]

    # 2) Normalize trial string
    rec_str = str(rec).zfill(3) if isinstance(rec, int) else rec

    # 3) Load all channel labels
    mat         = scipy.io.loadmat(os.path.join(MONKEYDIR, sess_day, "mat", "MRILabels.mat"))
    label_names = [lbl[0].lower() for lbl in mat["labelName"].flat]

    # 4) Match region key
    #    e.g. "(r) anterior amygdalar area" → "anterior amygdalar area"
    region_key = region_label.split(") ", 1)[1].lower()
    flags      = [region_key in lbl for lbl in label_names]
    chans      = [i + 1 for i, ok in enumerate(flags) if ok]

    if not chans:
        raise ValueError(f"No channels found for region “{region_key}” in session {sess_day!r}.")

    # 5) Load each channel via your existing loader
    data_list = [
        load_electrodes(pick_sess, rec_str, ch)
        for ch in chans
    ]
    # Stack into one array: (n_chans × n_samples)
    ch_data = np.vstack(data_list)

    print(f"Loaded {ch_data.shape[0]} channel(s) from region '{region_key}' "
          f"for trial {rec_str} in session {sess_day}.")
    return ch_data, sess_day

# --------------------------------------------------------------------------------
# Method 3: load all channels at once for a session & trial (NOT IMPLEMENTED)
# --------------------------------------------------------------------------------
def load_all_channels(pick_sess, rec):
    """
    Load LFP samples for all channels in a given session and trial.

    Parameters:
      pick_sess (int): index of session in movie database
      rec        (int or str): trial number or zero-padded string

    Returns:
      np.ndarray: 2D array of shape (n_samples, nCh)
    """
    # Load session info
    sessions = load_movie_database()
    sess = sessions[pick_sess]
    sess_day = sess['day']

    # Normalize trial string
    rec_str = str(rec).zfill(3) if isinstance(rec, int) else rec

    # File path (may need to double check the duplicated 126 situation)
    file_path = os.path.join(
        MONKEYDIR,
        sess_day,
        rec_str,
        f"rec{rec_str}.Frontal_1.lfp.dat"
    )

    # Read and reshape
    raw = np.fromfile(file_path, dtype=np.float32)
    try:
        data = raw.reshape(-1, nCh)
    except ValueError:
        raise IOError(f"File size does not match expected channels: {file_path}")

    print(f"Loaded all channels: data shape {data.shape}.")
    return data
    
# --------------------------------------------------------------------------------
# Read all user inputs and store them in a dataclass
# --------------------------------------------------------------------------------
@dataclass
class UserInput:
    """All user-selected parameters at the time of 'Load LFP'."""
    fs:          int
    session:     int
    trial:       int
    mode:        str
    region:      str
    channels:    List[int]
    bp_taps:     int
    bp_low:      int
    bp_high:     int
    rms_ms:      float
    outlier_z:   float
    threshold_z: float
    min_dur:     int
    mer_dur:     int
    window_size: int

def snapshot_inputs(window) -> UserInput:
    """Grab every relevant widget value and return a dataclass."""
    mode = window.mode_combo.currentText()

    if mode == "Select Channel(s)":
        channels = [window.channel_input.value()]
        region   = None
    elif mode == "Select Region(s)":
        region   = window.region_combo.currentText()
        channels = []                 # fill if you have a map region→channels
    else:  # "All Channels"
        channels = list(range(1, 221))  # 1-based indices
        region   = None

    return UserInput(
        session=window.session_input.value(),
        trial=window.trial_input.value(),
        mode=mode,
        channels=channels,
        region=region,
        bp_taps=window.bp_taps_spin.value(),
        bp_low=window.bp_low_spin.value(),
        bp_high=window.bp_high_spin.value(),
        rms_ms=window.rms_spin.value(),
        outlier_z=window.outlier.value(),
        threshold_z=window.lower_bound.value(),
        min_dur=window.min_dur_spin.value(),
        mer_dur=window.merge_dur_spin.value(),
        window_size=window.visual_size.value()
    )

