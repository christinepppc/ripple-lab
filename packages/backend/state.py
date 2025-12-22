# backend/state.py
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class SessionState:
    lfp: Optional[np.ndarray] = None
    fs: int = 1000
    det_res: Any = None
    norm_res: Any = None
    rej_res: Any = None
    avg_res: Any = None

STORE: Dict[str, SessionState] = {}