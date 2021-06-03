from PyQt5.QtWidgets import QApplication
from Player import *

import sys
import os
import time

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = Player()
    sys.exit(app.exec_())
