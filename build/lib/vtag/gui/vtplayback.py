import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui     import QPainter, QPen, QColor
from PyQt6.QtCore    import Qt

# vtag imports
from .colors        import palette_viridis

class VTPlayback(QWidget):

    def __init__(self):
        super().__init__()

        self.w = 0
        self.h = 0
        self.n = 0
        self.bin = 0
        self.ls_heat = []
        self.i_frame = 0
        self.i_frame_tmp = 0

        self.setMouseTracking(True)
        self.initUI()

    def initUI(self):
        self.setFixedHeight(30)
        self.repaint()

    def set_n(self, n):
        self.n = n
        self.repaint()

    def set_error(self, error):
        error = error.sum(axis=1)
        self.ls_heat = get_rank(error, len(palette_viridis))
        self.repaint()

    def set_heat(self, input):
        self.ls_heat = get_rank(input, len(palette_viridis))
        self.repaint()

    def set_frame(self, i):
        self.i_frame = i
        self.repaint()

    def set_frame_tmp(self, i):
        self.i_frame_tmp = i
        self.repaint()

    def enterEvent(self, evt):
        super().enterEvent(evt)
        self.i_frame_tmp = self.i_frame

    def paintEvent(self, evt):
        super().paintEvent(evt)
        self.w   = self.size().width()
        self.h   = self.size().height()
        self.bin = (self.w / self.n) if self.n != 0 else 1
        # plot color
        painter = QPainter(self)
        pen = QPen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setWidth(self.bin)
        base = int(self.bin / 2)
        for i in range(len(self.ls_heat)):
            rk = self.ls_heat[i]
            pen.setColor(palette_viridis[rk])
            painter.setPen(pen)
            pos = int(base + i * self.bin)
            painter.drawLine(pos, 0, pos, self.h)
        # current frame
        pen.setWidth(3)
        pen.setColor(QColor(255, 0, 0, 150))
        pos = int(base + self.i_frame * self.bin)
        painter.setPen(pen)
        painter.drawLine(pos, 0, pos, self.h)
        # tmp frame
        pen.setWidth(2)
        pen.setColor(QColor(255, 0, 0, 100))
        pos = int(base + self.i_frame_tmp * self.bin)
        painter.setPen(pen)
        painter.drawLine(pos, 0, pos, self.h)
        # border
        pen.setWidth(2)
        pen.setColor(QColor(255, 255, 0))
        painter.drawLine(0, 0, self.w, 0)
        painter.drawLine(self.w, 0, self.w, self.h)
        painter.drawLine(self.w, self.h, 0, self.h)
        painter.drawLine(0, self.h, 0, 0)

def draw_vline(painter, pos, height):
    painter.drawLine(pos, 0, pos, height)

def get_rank(input, n_bin):
    quans = np.quantile(np.unique(input[3:-3]), [i/n_bin for i in range(n_bin)])
    idx_q = []
    for i in input:
        val_abs = np.abs(i - quans)
        idx = np.where(val_abs == np.min(val_abs))[0][0]
        idx_q += [idx]
    return np.array(idx_q)
