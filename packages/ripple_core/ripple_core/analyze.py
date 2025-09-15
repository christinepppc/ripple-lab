import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from scipy.signal import firwin, filtfilt, find_peaks
from analysis.multitaper_spectrogram_python import multitaper_spectrogram

__all__ = [
    "DetectionResult",
    "detect_ripples",
    "NormalizationResult",
    "normalize_ripples",
    "RejectionResult",
    "reject_ripples",
]

# -----------------------------------------------------------------------------
# 1. Ripple Detection
# -----------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Container for all outputs produced by :func:`detect_ripples`."""

    bp_lfp: np.ndarray
    env_rip: np.ndarray
    bits: np.ndarray
    mu: float
    sd: float
    mu_og: float
    sd_og: float
    peak_idx: np.ndarray
    real_duration: np.ndarray
    windowed_duration: np.ndarray
    raw_windowed_lfp: np.ndarray
    bp_windowed_lfp: np.ndarray
    merged_starts: np.ndarray
    merged_ends: np.ndarray

def _rms_envelope(x: np.ndarray, win_len: int) -> np.ndarray:
    """Compute a simple sliding‐window RMS envelope."""
    kernel = np.ones(win_len) / win_len
    return np.sqrt(np.convolve(x ** 2, kernel, mode="same"))

def detect_ripples(
    lfp: np.ndarray,
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
) -> DetectionResult:
    """Detect candidate ripple events in an LFP trace.

    All numerical parameters are exposed as keyword‐only arguments so that the
    GUI can surface them to end users.  Defaults follow the original MATLAB
    implementation.
    """

    # ------------------------------------------------------------------
    # 1) Band‑pass filter 100–140 Hz (FIR, linear phase)
    # ------------------------------------------------------------------
    nyq = fs / 2
    b = firwin(order + 1, [rp_band[0] / nyq, rp_band[1] / nyq], pass_zero=False)
    bp_lfp = filtfilt(b, 1.0, lfp)

    # ------------------------------------------------------------------
    # 2) Sliding RMS envelope (+ smooth)
    # ------------------------------------------------------------------
    win_len = int(round(window_ms / 1000 * fs))
    env_rip = _rms_envelope(bp_lfp, win_len)
    # second pass – simple moving average for extra smoothing
    env_rip = np.convolve(env_rip, np.ones(win_len) / win_len, mode="same")

    # ------------------------------------------------------------------
    # 3) Two‑pass σ‑clipping threshold
    # ------------------------------------------------------------------
    mu_og, sd_og = np.mean(env_rip), np.std(env_rip)
    clipped = env_rip.copy()
    clipped[clipped > mu_og + z_outlier * sd_og] = np.nan
    mu, sd = np.nanmean(clipped), np.nanstd(clipped)
    bits = env_rip > mu + z_low * sd  # boolean mask of putative ripple periods

    # ------------------------------------------------------------------
    # 4) Duration cleanup (min length + merge short gaps)
    # ------------------------------------------------------------------
    min_samps = int(round(min_dur_ms / 1000 * fs))
    merge_samps = int(round(merge_dur_ms / 1000 * fs))

    starts, ends = [], []
    in_event = False
    for i, flag in enumerate(bits):
        if flag and not in_event:
            in_event = True
            cur_start = i
        if in_event and (not flag or i == len(bits) - 1):
            in_event = False
            cur_end = i if flag else i - 1
            if cur_end - cur_start + 1 >= min_samps:
                starts.append(cur_start)
                ends.append(cur_end)

    # Merge ripples closer than *merge_samps*
    merged_starts, merged_ends = [], []
    if starts:
        s, e = starts[0], ends[0]
        for s2, e2 in zip(starts[1:], ends[1:]):
            if s2 - e <= merge_samps:
                e = e2  # extend
            else:
                merged_starts.append(s)
                merged_ends.append(e)
                s, e = s2, e2
        merged_starts.append(s)
        merged_ends.append(e)
    merged_starts = np.asarray(merged_starts, dtype=int)
    merged_ends = np.asarray(merged_ends, dtype=int)

    # ------------------------------------------------------------------
    # 5) Center on peak & epoch (+/‑200 ms)
    # ------------------------------------------------------------------
    n_events = len(merged_starts)
    peak_idx = np.zeros(n_events, dtype=int)
    real_duration = np.column_stack((merged_starts, merged_ends))

    for k, (s, e) in enumerate(zip(merged_starts, merged_ends)):
        seg = bp_lfp[s : e + 1]
        peak_idx[k] = s + int(np.argmax(seg))

    epoch_samps = int(round(epoch_ms / 1000 * fs))
    total_samps = 2 * epoch_samps + 1  # 401 samples for fs=1 kHz, ±200 ms

    windowed_duration = np.empty((n_events, 2), dtype=int)
    raw_windowed_lfp = np.zeros((n_events, total_samps))
    bp_windowed_lfp = np.zeros((n_events, total_samps))

    for k, pk in enumerate(peak_idx):
        s_idx, e_idx = pk - epoch_samps, pk + epoch_samps
        windowed_duration[k] = (s_idx, e_idx)

        pad_pre = max(0, 1 - s_idx)
        pad_post = max(0, e_idx - (len(lfp) - 1))

        vs, ve = max(0, s_idx), min(e_idx, len(lfp) - 1)
        raw_seg = lfp[vs : ve + 1]
        bp_seg = bp_lfp[vs : ve + 1]

        if pad_pre:
            raw_seg = np.concatenate([np.zeros(pad_pre), raw_seg])
            bp_seg = np.concatenate([np.zeros(pad_pre), bp_seg])
        if pad_post:
            raw_seg = np.concatenate([raw_seg, np.zeros(pad_post)])
            bp_seg = np.concatenate([bp_seg, np.zeros(pad_post)])

        raw_windowed_lfp[k] = raw_seg[:total_samps]
        bp_windowed_lfp[k] = bp_seg[:total_samps]

    return DetectionResult(
        bp_lfp,
        env_rip,
        bits.astype(int),
        mu,
        sd,
        mu_og,
        sd_og,
        peak_idx,
        real_duration,
        windowed_duration,
        raw_windowed_lfp,
        bp_windowed_lfp,
        merged_starts,
        merged_ends,
    )


# -----------------------------------------------------------------------------
# 2. Normalisation (time‑frequency)
# -----------------------------------------------------------------------------
@dataclass
class NormalizationResult:
    """Outputs from :func:`normalize_ripples`."""

    spec_mu: np.ndarray
    spec_sig: np.ndarray
    normalized_ripple_windowed: np.ndarray  # (F × T × N)
    freq_spec_windowed: np.ndarray  # (N × F)
    freq_spec_actual: np.ndarray  # (N × F)
    spec_f: np.ndarray
    spec_t: np.ndarray

def normalize_ripples(
    lfp: np.ndarray,
    fs: int,
    raw_windowed_lfp: np.ndarray,
    real_duration: np.ndarray,
    *,
    fmin: int = 2,
    fmax: int = 200,
    win_length: float = 0.060,
    step: int = 0.001,
    nw: float = 1.2,
    tapers: int = 2, 
    tfspec_pad = 20
) -> NormalizationResult:
    
    """use the multitaper spectrogram method in substitution for tfspec
    """

    # Step 1: Find the baseline
    mt_spectrogram, stimes, sfreqs = multitaper_spectrogram(
        lfp,
        fs=1000,
        frequency_range=[fmin, fmax],
        time_bandwidth= nw,
        num_tapers = tapers,
        window_params = [win_length, step],
        plot_on = False,
        min_nfft = 256
    )

    spec_mu = mt_spectrogram.mean(axis=1, keepdims=True)
    spec_sig = mt_spectrogram.std(axis=1, ddof=1, keepdims=True)
    # Step 2: Initiate a structure to save down the time frequency matrix for each ripple
    n_events = raw_windowed_lfp.shape[0]
    mt_test, t_test, f_test = multitaper_spectrogram(
        raw_windowed_lfp[1],
        fs=1000,
        frequency_range=[fmin, fmax],
        time_bandwidth= nw,
        num_tapers = tapers,
        window_params = [win_length, step],
        plot_on = False,
        min_nfft = 256,
    )

    T_snip = len(t_test)
    F_snip = len(f_test)
    normalized_ripple_windowed = np.zeros((F_snip, T_snip, n_events))
    freq_spec_windowed = np.zeros((n_events, F_snip))
    freq_spec_actual = np.zeros((n_events, F_snip))

    # Step 3: Find and save spectrogram for the visualization purposes for each ripple
    for k in range(n_events):
        mt_spec_window, stimes, sfreqs = multitaper_spectrogram(
            raw_windowed_lfp[k],
            fs=1000,
            frequency_range=[fmin, fmax],
            time_bandwidth= nw,
            num_tapers = tapers,
            window_params = [win_length, step],
            plot_on = False,
            min_nfft = 256
        )

        norm_window = (mt_spec_window - spec_mu) / spec_sig
        normalized_ripple_windowed[:, :, k] = norm_window
        freq_spec_windowed[k] = norm_window.mean(axis=1)

        # Step 4: Find the actual ripple's spectrum for the rejection purposes(+/‑20 ms pad)
        rp_start, rp_end = real_duration[k]
        pad = int(round(tfspec_pad / 1000 * fs))
        s = max(0, rp_start - pad)
        e = min(len(lfp) - 1, rp_end + pad)
        seg = lfp[s : e + 1]
        mt_spec_ripseg, ripseg_t, ripseg_freq = multitaper_spectrogram(
            seg,
            fs=1000,
            frequency_range=[fmin, fmax],
            time_bandwidth= nw,
            num_tapers = tapers,
            window_params = [win_length, step],
            plot_on = False,
            min_nfft = 256
        )
        norm_ripple = (mt_spec_ripseg - spec_mu) / spec_sig
        freq_spec_actual[k] = norm_ripple.mean(axis=1)
 

    return NormalizationResult(
        spec_mu=spec_mu,
        spec_sig=spec_sig,
        normalized_ripple_windowed=normalized_ripple_windowed,
        freq_spec_windowed=freq_spec_windowed,
        freq_spec_actual=freq_spec_actual,
        spec_f= sfreqs,
        spec_t= stimes,
    )


# -----------------------------------------------------------------------------
# 3. Rejection logic
# -----------------------------------------------------------------------------
@dataclass
class RejectionResult:
    """Output from :func:`reject_ripples`."""
    pass_idx: List[int]
    reject_idx: List[int]
    reasons: List[str]
    markers: np.ndarray  # boolean mask (N,) – True if rejected
    counters: Dict[str, int]

def reject_ripples(
    freq_spec_actual: np.ndarray,
    spec_f: np.ndarray,
    mu: float,
    sd: float,
    strict_threshold: float,
    env_rip: np.ndarray,
    peak_idx: np.ndarray,
    *,
    noise_cut_off: float = 1e4,
    low_cut_freq: int = 30,
    rp_band: Tuple[int, int] = (100, 140),
    perc_height_cut_off: float = 0.5,
    height_cut_off: float = 1.0,
) -> RejectionResult:
    N = freq_spec_actual.shape[0]
    reject_idx, reasons, pass_idx = [], [], []
    counters = {
        "noise": 0,
        "multipeak": 0,
        "rank": 0,
        "double_high": 0,
        "out_band": 0,
    }

    # Boolean masks (constant w.r.t. event)
    band_mask = (spec_f >= rp_band[0]) & (spec_f <= rp_band[1])
    low_mask = (spec_f >= low_cut_freq) & (spec_f < rp_band[0])
    high_mask = spec_f > rp_band[1]
    full_mask = spec_f >= low_cut_freq
    outside_mask = low_mask | high_mask

    for q in range(N):
        Y = freq_spec_actual[q]
        if np.isnan(Y).all():
            reject_idx.append(q)
            reasons.append("nan")
            continue

        # Criterion 0 – excess broadband noise
        if Y.max() > noise_cut_off:
            counters["noise"] += 1
            reject_idx.append(q)
            reasons.append("noise")
            continue

        # Criterion 1 – unique peak inside ripple band
        pks, _ = find_peaks(Y[band_mask])
        if len(pks) != 1:
            counters["multipeak"] += 1
            reject_idx.append(q)
            reasons.append("multipeak")
            continue

        main_h = Y[band_mask][pks[0]]

        # Criterion 2 – main peak highest + most prominent across ≥low_cut_freq
        peaks_all, _ = find_peaks(Y[full_mask])
        if len(peaks_all) and (Y[full_mask][peaks_all].max() != main_h):
            counters["rank"] += 1
            reject_idx.append(q)
            reasons.append("rank")
            continue

        # Criterion 3 – two or more peaks above band comparable in height
        peaks_high, _ = find_peaks(Y[high_mask])
        if len(peaks_high) > 1 and Y[high_mask][peaks_high].max() >= perc_height_cut_off * main_h:
            counters["double_high"] += 1
            reject_idx.append(q)
            reasons.append("double_high")
            continue

        # Criterion 4 – peak outside band taller than main peak
        if Y[outside_mask].max() >= height_cut_off * main_h:
            counters["out_band"] += 1
            reject_idx.append(q)
            reasons.append("out_band")
            continue


        # Criterion 5 – main envelope peak must exceed μ + k·σ
        # (k == strict_threshold)
        strict_crit = mu + strict_threshold * sd

        # main peak amplitude from the event's ripple envelope
        if env_rip.ndim == 2:
            main_amp = env_rip[q, peak_idx[q]]
        else:  # env_rip is the current event vector
            main_amp = env_rip[peak_idx[q]]

        if (not np.isfinite(main_amp)) or (main_amp < strict_crit):
            counters["below_strict"] = counters.get("below_strict", 0) + 1
            reject_idx.append(q)
            reasons.append("below_strict")
            continue

        # Passed all checks
        pass_idx.append(q)

    markers = np.zeros(N, dtype=bool)
    markers[reject_idx] = True

    return RejectionResult(pass_idx, reject_idx, reasons, markers, counters)


# -----------------------------------------------------------------------------
# 4. Find Average of Detection Result logic
# -----------------------------------------------------------------------------

@dataclass
class GrandAverageResult:

    avg_lfp: np.ndarray
    avg_bp_lfp: np.ndarray
    avg_rej_lfp: np.ndarray
    avg_rej_bp_lfp: np.ndarray
    avg_tfspec: np.ndarray
    avg_rej_tfspec: np.ndarray
    avg_real_spectrum: np.ndarray
    avg_rej_real_spectrum: np.ndarray
    avg_visual_spectrum: np.ndarray
    avg_rej_visual_spectrum: np.ndarray

def find_avg(
    pass_idx: np.ndarray,
    reject_idx: np.ndarray,
    raw_windowed_lfp: np.ndarray,
    bp_windowed_lfp: np.ndarray,
    normalized_ripple_windowed: np.ndarray,
    freq_spec_windowed: np.ndarray,
    freq_spec_actual: np.ndarray
) -> GrandAverageResult:
    
    # find the lfp
    acc_raw_windowed_lfp = raw_windowed_lfp[pass_idx, :]
    avg_lfp = np.mean(acc_raw_windowed_lfp , axis=0)
    rej_raw_windowed_lfp = raw_windowed_lfp[reject_idx, :]
    avg_rej_lfp = np.mean(rej_raw_windowed_lfp , axis=0)

    # find the bandpass ripple-band lfp
    acc_bp_windowed_lfp = bp_windowed_lfp[pass_idx, :]
    avg_bp_lfp = np.mean(acc_bp_windowed_lfp , axis=0)
    rej_bp_windowed_lfp = bp_windowed_lfp[reject_idx, :]
    avg_rej_bp_lfp = np.mean(rej_bp_windowed_lfp , axis=0)

    # find the spectrogram
    acc_tfspec = normalized_ripple_windowed[:, :, pass_idx]
    avg_tfspec = np.mean(acc_tfspec, axis=2)
    rej_tfspec = normalized_ripple_windowed[:, :, reject_idx]
    avg_rej_tfspec = np.mean(rej_tfspec, axis=2)

    # find the real spectrum
    acc_real_spectrum = freq_spec_actual[pass_idx, :]
    avg_real_spectrum = np.mean(acc_real_spectrum, axis=1)
    rej_real_spectrum = freq_spec_actual[reject_idx, :]
    avg_rej_real_spectrum = np.mean(rej_real_spectrum, axis=1)

    # find the windowed/visualization spectrum
    acc_visual_spectrum = freq_spec_windowed[:, pass_idx]
    avg_visual_spectrum = np.mean(acc_visual_spectrum, axis=1)
    rej_visual_spectrum = freq_spec_windowed[:, reject_idx]
    avg_rej_visual_spectrum = np.mean(rej_visual_spectrum, axis=1)

    return GrandAverageResult(
        avg_lfp,
        avg_bp_lfp,
        avg_rej_lfp,
        avg_rej_bp_lfp,
        avg_tfspec,
        avg_rej_tfspec,
        avg_real_spectrum,
        avg_rej_real_spectrum,
        avg_visual_spectrum,
        avg_rej_visual_spectrum)