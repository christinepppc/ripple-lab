# packages/ripple_core/ripple_core/signal/__init__.py

from .bipolar import (
    make_bipolar_pairs_from_grid,
    process_bipolar_referencing,
    detect_ripples_on_bipolar_channels,
    WHITE_MATTER_CHANNELS
)

__all__ = [
    "make_bipolar_pairs_from_grid",
    "process_bipolar_referencing",
    "detect_ripples_on_bipolar_channels",
    "WHITE_MATTER_CHANNELS",
]
