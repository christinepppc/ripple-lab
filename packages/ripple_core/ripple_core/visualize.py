# analysis/visualize.py
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ripple_core.analyze import GrandAverageResult
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QPixmap, QImage, QColor
import numpy as np
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from scipy.signal import welch
from scipy.signal.windows import dpss

# Tab 0: Show users the steps when they're processing ripples
def _downsample_for_display(t, y, max_pts=200_000):
    """Keep plotting responsive for long recordings."""
    n = y.size
    if n <= max_pts:
        return t, y
    step = int(np.ceil(n / max_pts))
    return t[::step], y[::step]

def _plot_vlines(axs, times, color, alpha=0.25, lw=0.6):
    """Draw the same vertical lines on multiple axes."""
    if times is None or len(times) == 0:
        return
    for ax in axs:
        for tt in times:
            ax.axvline(tt, color=color, alpha=alpha, lw=lw)

def _mtm_psd_db(x: np.ndarray, fs: float, NW: float = 3.0, nfft: int = 512):
    """
    Minimal multitaper PSD (PMTM-style):
      - DPSS tapers with time-bandwidth NW
      - K = 2*NW - 1 tapers (integer)
      - rFFT magnitude-squared, averaged across tapers
      - Returns (f, 10*log10(PSD))
    """
    x = np.asarray(x, dtype=float)
    K = max(1, int(2 * NW - 1))
    # center the signal (common for PSD)
    x = x - np.mean(x)

    # DPSS tapers: shape (K, N)
    tapers = dpss(M=x.size, NW=NW, Kmax=K, sym=False)  # (K, N)

    # FFT frequency axis (one-sided)
    f = np.fft.rfftfreq(nfft, d=1.0/fs)

    # accumulate power across tapers
    P = 0.0
    for k in range(K):
        xt = x * tapers[k]
        Xf = np.fft.rfft(xt, n=nfft)
        P += (np.abs(Xf) ** 2)

    P /= K  # average across tapers

    # scale to PSD per Hz (simple scaling; fine for comparison/plotting)
    # Normalize by sampling freq and window length to approximate density
    P = P / (fs * x.size)

    P_db = 10.0 * np.log10(np.maximum(P, np.finfo(float).tiny))
    return f, P_db

def plot_processing_notebook(
    fig: Figure,
    lfp: np.ndarray,
    fs: float,
    *,
    bp_lfp: np.ndarray | None = None,
    env_rip: np.ndarray | None = None,
    event_bounds: np.ndarray | None = None,  # (N,2) samples
    fmax: float = 200.0,
    t0_sec: float = 0.0,          # <-- NEW: window start (seconds)
    row_heights=(1.3, 1.3, 1.3, 3.3),   # <-- make row 4 ~3.5x taller
    duration_sec: float = 60.0,   # <-- NEW: show first 60 s by default
):
    fig.clear()
    gs  = fig.add_gridspec(4, 1, height_ratios=row_heights, hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])  # spectra

    # ---- compute crop indices for time window (rows 1–3) ----------
    n = lfp.size
    i0 = max(0, int(round(t0_sec * fs)))
    i1 = min(n, int(round((t0_sec + duration_sec) * fs)))
    if i1 <= i0:
        i1 = min(n, i0 + int(round(1 * fs)))  # guarantee >=1s

    # crop indices
    n   = lfp.size
    i0  = max(0, int(round(t0_sec * fs)))
    i1  = min(n, int(round((t0_sec + duration_sec) * fs)))
    if i1 <= i0:
        i1 = min(n, i0 + int(round(1 * fs)))

    # RELATIVE time for the visible window: 0 … duration_sec
    t_rel = (np.arange(i0, i1) / fs) - t0_sec   # <- left edge is exactly 0
    x0, x1 = 0.0, duration_sec

    # Row 1: raw LFP (cropped)
    y1 = lfp[i0:i1]
    t1, y1 = _downsample_for_display(t_rel, y1)
    ax1.plot(t_rel, y1, lw=0.6, color="#1f77b4")    
    ax1.set_ylabel("μV", fontsize=9)
    ax1.set_title(f"Raw LFP  (Only showing first 60 s)", fontsize=10)
    ax1.yaxis.set_major_locator(MaxNLocator(3))
    ax1.tick_params(labelsize=8)
    ax1.grid(alpha=0.2, linestyle=":", linewidth=0.5)

    # Row 2: band-pass LFP (cropped if provided)
    if bp_lfp is not None:
        y2 = bp_lfp[i0:i1]
        t2, y2 = _downsample_for_display(t_rel, y2)
        ax2.plot(t_rel, y2, lw=0.6, color="#d62728")
        ax2.set_ylabel("μV", fontsize=9)
        ax2.set_title("Band-passed LFP", fontsize=10)
        ax2.yaxis.set_major_locator(MaxNLocator(3))
        ax2.tick_params(labelsize=8)
        ax2.grid(alpha=0.2, linestyle=":", linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "Run Detect Ripples to view\nband-passed signal",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        ax2.set_yticks([])

    # Row 3: RMS envelope (cropped) + event lines within window
    if env_rip is not None:
        e3 = env_rip[i0:i1]
        t3, e3 = _downsample_for_display(t_rel, e3)
        ax3.plot(t_rel, e3, lw=0.7, color="#2ca02c")
        ax3.set_ylabel("RMS (μV)", fontsize=9)
        ax3.set_title("RMS Envelope", fontsize=10)
        ax3.yaxis.set_major_locator(MaxNLocator(3))
        ax3.tick_params(labelsize=8)
        ax3.grid(alpha=0.2, linestyle=":", linewidth=0.5)
    
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(x0, x1)
        ax.margins(x=0)  # no left/right padding

    if event_bounds is not None and event_bounds.size:
        on_times  = event_bounds[:, 0] / fs
        off_times = event_bounds[:, 1] / fs
        # mask events that fall within the visible absolute window
        abs_start, abs_end = t0_sec, t0_sec + duration_sec
        mask_on  = (on_times  >= abs_start) & (on_times  <= abs_end)
        mask_off = (off_times >= abs_start) & (off_times <= abs_end)
        on_rel   = on_times[mask_on]  - t0_sec
        off_rel  = off_times[mask_off] - t0_sec
        for ax in (ax1, ax2, ax3):
            for tt in on_rel:
                ax.axvline(tt, color="tab:green", alpha=0.25, lw=0.6)
            for tt in off_rel:
                ax.axvline(tt, color="tab:blue",  alpha=0.25, lw=0.6)

    for ax in (ax1, ax2, ax3):
        ax.set_xticks([0, duration_sec/2, duration_sec])
    ax3.set_xlabel("Time (s)", fontsize=9)


    # Row 4: spectra (unchanged; still full recording)
    try:
        f_raw, Pxx_raw_db = _mtm_psd_db(lfp, fs=fs, NW=3.0, nfft=512)
        ax4.plot(f_raw, Pxx_raw_db, lw=0.9, color="blue", label="LFP Original Signal")
        if bp_lfp is not None:
            f_bp, Pxx_bp_db = _mtm_psd_db(bp_lfp, fs=fs, NW=3.0, nfft=512)
            ax4.plot(f_bp, Pxx_bp_db, lw=0.9, color=(1.0, 0.5, 0.0), label="Bandpass Signal")
        ax4.set_xlim(0, fs / 2)
        ax4.set_xlabel("Frequency (Hz)", fontsize=9)
        ax4.set_ylabel("Power (dB-converted)", fontsize=9)
        ax4.yaxis.set_major_locator(MaxNLocator(3))
        ax4.tick_params(labelsize=8)
        ax4.grid(alpha=0.2, linestyle=":", linewidth=0.5)
        ax4.legend(loc="upper right", fontsize=8, frameon=False)
        ax4.set_title("Bandpass vs. LFP Original Spectrum  |  nfft: 512; NW = 3", fontsize=10)
    except Exception:
        ax4.text(0.5, 0.5, "Unable to compute spectra", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=10)
        ax4.set_xticks([]); ax4.set_yticks([])
 
# Tab 1: Show grand-average summary for one session one trial
def plot_grand_average_grid(
    grand:    GrandAverageResult,
    spec_f:   np.ndarray,
    fig:      Figure,
    canvas,                      # FigureCanvasQTAgg
    *,
    epoch_ms: int,
    session:  int,
    trial:    int,
    channel:  int,
    mode:     str,
    fmin:     int,
    fmax:     int
):
    """
    3 × 2 grand-average layout:

        Row 0  (Raw)     accepted | rejected
        Row 1  (BP)      accepted | rejected
        Row 2  (Spec)    accepted | rejected
    """

    fig.clear()
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        height_ratios=[1, 1, 3],
        hspace=0.40,
        wspace=0.20,
        figure=fig,
    )

    t = np.linspace(-epoch_ms, epoch_ms, grand.avg_lfp.size)

    # ---------------- Row 0 – raw LFP ----------------------------------
    ax_raw_acc = fig.add_subplot(gs[0, 0])
    ax_raw_rej = fig.add_subplot(gs[0, 1])

    ax_raw_acc.plot(t, grand.avg_lfp, color="g")
    ax_raw_acc.set_title("Raw LFP – Accepted")
    ax_raw_acc.set_ylabel("µV")
    ax_raw_acc.set_xlim(t[0], t[-1])

    ax_raw_rej.plot(t, grand.avg_rej_lfp, color="r")
    ax_raw_rej.set_title("Raw LFP – Rejected")
    ax_raw_rej.set_xlim(t[0], t[-1])

    # ---------------- Row 1 – BP LFP -----------------------------------
    ax_bp_acc = fig.add_subplot(gs[1, 0])
    ax_bp_rej = fig.add_subplot(gs[1, 1])

    ax_bp_acc.plot(t, grand.avg_bp_lfp, color="g")
    ax_bp_acc.set_title(f"Band-pass {fmin:d}–{fmax:d} Hz — Accepted")
    ax_bp_acc.set_ylabel("µV")
    ax_bp_acc.set_xlim(t[0], t[-1])

    ax_bp_rej.plot(t, grand.avg_rej_bp_lfp, color="r")
    ax_bp_rej.set_title(f"Band-pass {fmin:d}–{fmax:d} Hz — Rejected")
    ax_bp_rej.set_xlim(t[0], t[-1])

    # ---------------- Row 2 – spectrograms -----------------------------
    extent = [-epoch_ms, epoch_ms, spec_f[0], spec_f[-1]]
    yticks = np.arange(fmin, fmax, np.size(spec_f))  # inclusive

    ax_spec_acc = fig.add_subplot(gs[2, 0])
    im_acc = ax_spec_acc.imshow(
        grand.avg_tfspec,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="hot",
    )
    ax_spec_acc.set_title("Grand Average Spectrogram – Accepted")
    ax_spec_acc.set_ylim(fmin, fmax)
    ax_spec_acc.set_yticks(yticks)
    ax_spec_acc.set_ylabel("Frequency (Hz)")
    ax_spec_acc.set_xlabel("Time (ms)")
    ax_spec_acc.set_ylabel("Freq (Hz)")
    fig.colorbar(im_acc, ax=ax_spec_acc, fraction=0.046, pad=0.02)

    ax_spec_rej = fig.add_subplot(gs[2, 1])
    im_rej = ax_spec_rej.imshow(
        grand.avg_rej_tfspec,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="hot",
    )
    ax_spec_rej.set_title("Grand Average Spectrogram – Rejected")
    ax_spec_rej.set_xlabel("Time (ms)")
    ax_spec_rej.set_ylim(fmin, fmax)
    ax_spec_rej.set_yticks(yticks)
    ax_spec_rej.set_ylabel("Frequency (Hz)")
    fig.colorbar(im_rej, ax=ax_spec_rej, fraction=0.046, pad=0.02)

    # ---------------- Global title -------------------------------------
    fig.suptitle(
        f"Session {session}  |  Trial {trial}  |  Channel {channel} | {mode}",
        fontsize=14,
        fontweight="bold",
    )

    canvas.draw()
    return fig

# Tab 2 Helpers New
def make_lfp_pix(
    trace: np.ndarray,
    color: str = "#1f77b4",
    w: int     = 300,     # wider so ticks fit
    h: int     = 200,
    epoch_ms: int = 200,
) -> QPixmap:
    """
    Raw or BP trace → QPixmap *with* x/y axes.
    """
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    cvs = FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)
    t   = np.linspace(-epoch_ms, epoch_ms, trace.size)
    ax.plot(t, trace, color=color, lw=0.8)

    # --- axes cosmetics --------------------------------------------
    ax.set_xlim(-epoch_ms,epoch_ms)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("ms", fontsize=6)
    ax.set_ylabel("$\mu V$'", fontsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=6)
    ax.set_frame_on(False)
    fig.tight_layout(pad=0.2)

    cvs.draw(); buf = cvs.buffer_rgba()
    return QPixmap.fromImage(QImage(buf, w, h, QImage.Format_ARGB32))

def make_spectrum_pix(
    trace: np.ndarray,
    color: str = "#1f77b4",
    w: int     = 300,     # wider so ticks fit
    h: int     = 200,
    fmin: int  = 2,
    fmax: int  = 200,
) -> QPixmap:

    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    cvs = FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)
    freq   = np.linspace(0, 200, trace.size)  # ms
    ax.plot(freq, trace, color=color, lw=0.8)

    # --- axes cosmetics --------------------------------------------
    ax.set_xlim(fmin, fmax)
    ax.xaxis.set_major_locator(FixedLocator([fmin,100,140,fmax]))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{fmin}","100","140",f"{fmax}"]))
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("Hz", fontsize=6)
    ax.set_ylabel("Power ($\mu V^2$')", fontsize=6)
    
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=6)
    ax.set_frame_on(False)
    fig.tight_layout(pad=0.2)

    cvs.draw(); buf = cvs.buffer_rgba()
    return QPixmap.fromImage(QImage(buf, w, h, QImage.Format_ARGB32))

def make_spec_pix(
    S: np.ndarray,
    cmap: str = "hot",
    faxis: np.ndarray | None = None,  # optional freq vector
    w: int = 300,
    h: int = 200,
    epoch_ms: int = 200,
    fmin: int = 2,
    fmax: int = 200
) -> QPixmap:
    """
    Spectrogram → QPixmap with freq/time axes.
    """
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    cvs = FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)
    im = ax.imshow(S, aspect="auto", origin="lower", cmap=cmap)
    # freq axis ticks (0,100,200 Hz or from faxis)
    if faxis is None:
        nF = S.shape[0]
        faxis = np.linspace(fmin, fmax, nF)

    ax.set_yticks([0, len(faxis)//2, len(faxis)-1])
    ax.set_yticklabels([f"{faxis[0]:.0f}", f"{faxis[len(faxis)//2]:.0f}", f"{faxis[-1]:.0f}"],
                       fontsize=6)
    ax.set_xticks([0, S.shape[1]//2, S.shape[1]-1])
    ax.set_xticklabels([
        f"-{epoch_ms}",
        "0",
        f"{epoch_ms}"
    ], fontsize=6)
    ax.set_xlabel("ms", fontsize=6)
    ax.set_ylabel("Hz", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Power ($\mu V^2$)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    ax.set_frame_on(False)
    fig.tight_layout(pad=0.2)

    cvs.draw(); buf = cvs.buffer_rgba()
    return QPixmap.fromImage(QImage(buf, w, h, QImage.Format_RGBA8888))