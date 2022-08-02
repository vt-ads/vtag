import sys
from PyQt6.QtWidgets import QApplication
from .gui.vtgui    import VTGUI

args = sys.argv
app  = QApplication(args)
gui  = VTGUI(args)
sys.exit(app.exec())
