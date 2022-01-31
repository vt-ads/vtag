from PyQt5.QtWidgets import QApplication
from GUI_Player import *

if __name__ == '__main__':
    args   = sys.argv
    app    = QApplication(args)
    player = VTPlayer(args)
    sys.exit(app.exec_())
