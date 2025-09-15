# ripple_core/signal/reref.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict
import numpy as np
import scipy.io as sio

from .pairs import make_bipolar_pairs_from_grid

def _find_single_mat(ch_dir: Path) -> Path | None:
    """Find exactly one .mat under chan### (prefer lfp.mat)."""
    cand = list(ch_dir.glob("lfp.mat"))
    if len(cand) == 1:
        return cand[0]
    # fallback: any single .mat
    cand = list(ch_dir.glob("*.mat"))
    if len(cand) == 1:
        return cand[0]
    return None

def _load_lfp(mat_path: Path) -> tuple[np.ndarray, float | None]:
    S = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "lfp" not in S:
        raise KeyError(f"'lfp' missing in {mat_path}")
    x = np.asarray(S["lfp"]).astype(np.float32).ravel()
    fs = None
    if "fs" in S:
        try:
            fs = float(np.asarray(S["fs"]).squeeze())
        except Exception:
            fs = None
    return x, fs

def reref_trial(rootDir: str | Path, badCh: Iterable[int] = (), prefer="horizontal") -> Dict:
    """
    Mirror MATLAB reref_trial:
      - discover chan### with a single .mat each (containing lfp [+ fs])
      - compute y = x(i) - x(j) for neighbor pairs (within-bank)
      - first horizontal, then vertical ONLY for anchors not already written
      - write to <rootDir>_re-referenced/chan###_ref/lfp_ref.mat
      - also write pairs_used.mat with the list of (i,j) produced
    Returns a summary dict with counts and output path.
    """
    root = Path(rootDir)
    assert root.is_dir(), f"Root not found: {root}"
    out_dir = root.with_name(root.name + "_re-referenced")
    out_dir.mkdir(exist_ok=True)

    # discover available channels
    avail = np.zeros(221, dtype=bool)  # 1..220
    chan_path = [None]*221
    for ch in range(1, 221):
        ch_dir = root / f"chan{ch:03d}"
        if not ch_dir.is_dir():
            continue
        f = _find_single_mat(ch_dir)
        if f is None:
            continue
        avail[ch] = True
        chan_path[ch] = f

    # build pairs
    pairs_h, pairs_v = make_bipolar_pairs_from_grid(badCh)
    order = ["horizontal","vertical"]
    if prefer == "vertical":
        order = ["vertical","horizontal"]

    pairs_used = []
    anchor_done = np.zeros(221, dtype=bool)  # mirrors MATLAB anchor guard

    def _process_pairs(pairs: np.ndarray, tag: str) -> int:
        n_saved = 0
        for i, j in pairs:
            if anchor_done[i]:
                # skip immediately if anchor already done
                continue
            if not (avail[i] and avail[j]):
                continue
            fi = chan_path[i]; fj = chan_path[j]
            if fi is None or fj is None:
                continue

            xi, fs_i = _load_lfp(fi)
            xj, fs_j = _load_lfp(fj)
            T = min(xi.size, xj.size)
            if T == 0:
                continue
            lfp_ref = (xi[:T] - xj[:T]).astype(np.float32)

            fs = fs_i if fs_i is not None else (fs_j if fs_j is not None else np.nan)
            pair = np.array([i, j], dtype=np.int32)
            note = f"bipolar neighbor ({tag}): ch{i} - ch{j} (within bank)"

            out_ch_dir = out_dir / f"chan{i:03d}_ref"
            out_ch_dir.mkdir(parents=True, exist_ok=True)
            sio.savemat(out_ch_dir / "lfp_ref.mat", {
                "lfp_ref": lfp_ref,
                "fs": fs,
                "pair": pair,
                "note": note
            }, do_compression=True)

            anchor_done[i] = True  # crucial guard, like MATLAB
            pairs_used.append([i, j])
            n_saved += 1
        return n_saved

    counts = {}
    for tag in order:
        pairs = pairs_h if tag == "horizontal" else pairs_v
        counts[tag] = _process_pairs(pairs, tag)

    # manifest
    if pairs_used:
        sio.savemat(out_dir / "pairs_used.mat", {"pairs_used": np.asarray(pairs_used, dtype=np.int32)})

    # also mark any existing chan???_ref from previous runs as done (optional)
    for d in out_dir.glob("chan???_ref"):
        try:
            i = int(d.name[4:7])
            anchor_done[i] = True
        except Exception:
            pass

    return {
        "out_dir": str(out_dir),
        "n_horizontal": int(counts.get("horizontal", 0)),
        "n_vertical": int(counts.get("vertical", 0)),
        "n_total": int(sum(counts.values())),
    }