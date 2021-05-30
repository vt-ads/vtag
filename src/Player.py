from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget, QLabel, QSlider,
                             QGridLayout,  QVBoxLayout, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal

import sys, os, time
import cv2 as cv

class Player(QWidget):
    def __init__(self):
        super().__init__()

        # WD
        folder = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/one_pig"
        os.chdir(folder)

        # Frames
        self.paths   = get_imgs_path(folder)
        self.frame   = QLabel(self)
        self.n_frame = len(self.paths)
        self.i_frame = 0
        self.fps     = 10 / 1000

        # Status
        self.i_frame = 0
        self.is_play = False

        # Setup timer
        self.timer  = QTimer(self)

        # GUI
        self.buttons = dict(browse= QPushButton("Browse"),
                            play  = QPushButton("Play"),
                            next  = QPushButton("Next frame > "),
                            prev  = QPushButton("< Previous frame"))

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_frame - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.setTickInterval(1)

        # init
        self.update_frames()
        self.initRuntime()
        self.initUI()

    def initRuntime(self):
        self.timer.timeout.connect(self.next_frames)
        self.buttons["play"].clicked.connect(
            lambda x: self.change_status(not self.is_play))
        self.buttons["next"].clicked.connect(self.next_frames)
        self.buttons["prev"].clicked.connect(self.prev_frames)
        self.slider.valueChanged.connect(self.traverse_frames)

    def initUI(self):
        layout = QGridLayout(self)
        self.frame.setSizePolicy(QSizePolicy.Maximum,
                                 QSizePolicy.Maximum)
        self.frame.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.buttons["browse"], 0, 0, 1, 3)
        layout.addWidget(self.frame,             1, 0, 1, 3, alignment=Qt.AlignCenter)
        layout.addWidget(self.slider,            2, 0, 1, 3)
        layout.addWidget(self.buttons["prev"],   3, 0)
        layout.addWidget(self.buttons["play"],   3, 1)
        layout.addWidget(self.buttons["next"],   3, 2)
        self.setLayout(layout)

        self.move(300, 200)
        self.setWindowTitle('Virtual Tags')
        self.setGeometry(50, 50, 320, 200)

        self.show()

    def next_frames(self):
        self.i_frame += 1
        if self.i_frame == self.n_frame:
            self.change_status(to_play=False)
            self.i_frame = 0

        self.update_frames()

    def prev_frames(self):
        self.i_frame -= 1
        self.update_frames()

    def update_frames(self):
        i = self.i_frame
        self.slider.setValue(i)
        self.frame.setPixmap(QPixmap(self.paths[i]))

    def traverse_frames(self):
        self.i_frame = self.slider.value()
        self.update_frames()

    def change_status(self, to_play):
        if to_play:
            self.is_play = True
            self.buttons["play"].setText("Pause")
            self.timer.start(int(1 / self.fps))

        else:
            self.is_play = False
            self.buttons["play"].setText("Play")
            self.timer.stop()


def get_imgs_path(path):
    ls_imgs = os.listdir(path)
    ls_imgs.sort()
    return ls_imgs


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = Player()
    sys.exit(app.exec_())



# Note
# self.thread = QThread(self)
# self.thread.started.connect(self.update_image)
