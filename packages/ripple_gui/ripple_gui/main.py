# main.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon

ASSETS = Path(__file__).parent / "assets"

def main():
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