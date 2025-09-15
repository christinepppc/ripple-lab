# ripple_core/signal/pairs.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np

# Exact 26 x 10 grid from your MATLAB
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

def _bank_of(ch: int) -> int:
    if   1  <= ch <=  32: return 1
    elif 33 <= ch <=  64: return 2
    elif 65 <= ch <=  96: return 3
    elif 97 <= ch <= 128: return 4
    elif 129<= ch <= 160: return 5
    elif 161<= ch <= 192: return 6
    elif 193<= ch <= 220: return 7
    return -1

def make_bipolar_pairs_from_grid(badCh: Iterable[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recreates MATLAB's make_bipolar_pairs_from_grid.
    Returns:
      pairs_h: (K_h, 2) i j with j = right neighbor  (i - j)
      pairs_v: (K_v, 2) i j with j = below neighbor  (i - j)
    Applies: same-bank constraint; skips zeros and bad channels.
    """
    G = _LAYOUT_26x10
    bad = set(int(b) for b in badCh)
    R, C = G.shape
    pairs_h: List[tuple[int,int]] = []
    pairs_v: List[tuple[int,int]] = []

    # horizontal
    for r in range(R):
        for c in range(C-1):
            i, j = int(G[r,c]), int(G[r,c+1])
            if i==0 or j==0:         continue
            if i in bad or j in bad: continue
            if _bank_of(i) != _bank_of(j): continue
            pairs_h.append((i,j))

    # vertical
    for r in range(R-1):
        for c in range(C):
            i, j = int(G[r,c]), int(G[r+1,c])
            if i==0 or j==0:         continue
            if i in bad or j in bad: continue
            if _bank_of(i) != _bank_of(j): continue
            pairs_v.append((i,j))

    return np.asarray(pairs_h, dtype=int), np.asarray(pairs_v, dtype=int)