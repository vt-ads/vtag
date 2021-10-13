from PyQt5.QtWidgets import QApplication
from Player import *

if __name__ == '__main__':
    args   = sys.argv
    app    = QApplication(args)
    player = Player(args)
    sys.exit(app.exec_())
