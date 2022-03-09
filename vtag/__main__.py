import sys
from PyQt6.QtWidgets import QApplication
from .gui.vtplayer    import VTPlayer

args   = sys.argv
app    = QApplication(args)
player = VTPlayer(args)
sys.exit(app.exec())
