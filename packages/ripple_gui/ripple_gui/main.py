# main.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon

ASSETS = Path(__file__).parent / "assets"

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ASSETS / "logo.png")))   # global app icon

    from gui.windows import MainWindow

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