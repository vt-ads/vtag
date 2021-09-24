from lib import *

alpha = 200
palette_viridis = [
    QColor(68, 1, 84, alpha),
    QColor(72, 40, 62, alpha),
    QColor(62, 74, 137, alpha),
    QColor(49, 104, 142, alpha),
    QColor(38, 130, 142, alpha),
    QColor(31, 158, 137, alpha),
    QColor(53, 183, 121, alpha),
    QColor(109, 205, 89, alpha),
    QColor(180, 222, 44, alpha),
    QColor(253, 231, 37, alpha)
]
palette_viridis.reverse()

class Playback(QWidget):

    def __init__(self):
        super().__init__()

        self.w = 0
        self.h = 0
        self.n = 0
        self.bin = 0
        self.ls_heat = []
        self.i_frame = 0

        self.setMouseTracking(True)
        self.initUI()

    def initUI(self):
        self.setMinimumHeight(50)
        self.repaint()

    def set_n(self, n):
        self.n = n

    def set_heat(self, input):
        self.ls_heat = get_rank(input, len(palette_viridis))
        self.repaint()

    def set_frame(self, i):
        self.i_frame = i
        self.repaint()

    def paintEvent(self, evt):
        super().paintEvent(evt)
        self.w = self.size().width()
        self.h = self.size().height()
        self.bin = (self.w / self.n)
        # plot color
        painter = QPainter(self)
        pen = QPen()
        pen.setStyle(Qt.SolidLine)
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
        pen.setColor(QColor(255, 0, 0))
        pos = int(base + self.i_frame * self.bin)
        painter.setPen(pen)
        painter.drawLine(pos, 0, pos, self.h)
        # border
        pen.setWidth(2)
        pen.setColor(QColor(255, 255, 0))
        painter.drawLine(0, 0, self.w, 0)
        painter.drawLine(self.w, 0, self.w, self.h)
        painter.drawLine(self.w, self.h, 0, self.h)
        painter.drawLine(self.h, 0, 0, 0)

def draw_vline(painter, pos, height):
    painter.drawLine(pos, 0, pos, height)

def get_rank(input, n_bin):
    quans = np.quantile(input[3:-3], [i/n_bin for i in range(n_bin)])
    idx_q = []
    for i in input:
        val_abs = np.abs(i - quans)
        idx = np.where(val_abs == np.min(val_abs))[0][0]
        idx_q += [idx]
    return np.array(idx_q)
