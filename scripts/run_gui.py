#!/usr/bin/env python3
"""
Convenience launcher for the PyQt GUI from the repo root.

Usage:
  python3 scripts/run_gui.py
"""

from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    gui_pkg_root = repo_root / "packages" / "ripple_gui"
    sys.path.insert(0, str(gui_pkg_root))
    from ripple_gui.main import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()

