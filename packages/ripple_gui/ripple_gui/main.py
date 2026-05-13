import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon

ASSETS = Path(__file__).parent / "assets"

def _has_gui_display() -> bool:
    # If a non-xcb Qt platform plugin is explicitly requested (e.g. offscreen/vnc),
    # allow startup even without DISPLAY/WAYLAND_DISPLAY.
    platform = (os.environ.get("QT_QPA_PLATFORM") or "").strip().lower()
    if platform and platform != "xcb":
        return True
    # Otherwise require X11 (DISPLAY) or Wayland (WAYLAND_DISPLAY).
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

def main():
    if not _has_gui_display():
        raise SystemExit(
            "No GUI display detected (DISPLAY/WAYLAND_DISPLAY not set).\n"
            "To run the GUI, launch from a machine with a desktop session, or use X11 forwarding:\n"
            "  ssh -X <host>\n"
            "Then run:\n"
            "  python3 -m ripple_gui.main\n"
            "If you only need a headless run, you can set:\n"
            "  QT_QPA_PLATFORM=offscreen\n"
            "To launch with a remote-viewable display, you can try:\n"
            "  QT_QPA_PLATFORM=vnc\n"
        )
    app = QApplication(sys.argv)
    # Check if logo exists before setting it
    logo_path = ASSETS / "logo.png"
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))   # global app icon

    from ripple_gui.windows import MainWindow

    # Load the QSS stylesheet
    qss_path = Path(__file__).parent / "styles.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    # Launch main window
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()