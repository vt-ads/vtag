from lib import *
from Tags import VTags

colorsets = np.array(["#000000",
                      "#ffff33", "#f94144", "#f3722c", "#f8961e", "#f9844a",
                      "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590",
                      "#277da1"])

class Player(QWidget):
    def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data"):
        super().__init__()

        # WD
        dataname = "one_pig_small"
        os.chdir(folder)
        os.chdir(dataname)

        # Predictions
        self.imgs_show = []
        # Frames
        self.folder  = folder
        self.paths   = ls_files()
        self.frame   = QFrame()
        self.plot    = pg.plot()
        self.n_frame = len(self.paths)
        self.i_frame = 0
        self.lb_frame = QLabel("Frame: %d" % self.i_frame)
        self.fps     = 12.5 / 1000

        # Predictions
        self.sli_thre = QSlider(Qt.Horizontal, self)
        self.lb_thre  = QLabel("Threshold: %.3f" % (self.sli_thre.value()/1000))
        self.sli_span = QSlider(Qt.Horizontal, self)
        self.lb_span  = QLabel("Span: %d" % self.sli_span.value())

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
        self.toggles = dict(edges = QRadioButton("Edges"),
                            cls   = QRadioButton("Clusters"),
                            pre   = QRadioButton("Predictions"))
        self.playback = QSlider(Qt.Horizontal, self)

        # VTags
        self.app = VTags()

        # init
        self.init_VTags()
        self.update_frames()
        self.initRuntime()
        self.initUI()

    def init_VTags(self):
        self.app.load(h5="model_24f_20t.h5")
        self.toggles["pre"].setChecked(True)
        self.imgs_show = self.app.OUTS["pred"] ### define what show on the screen

    def initRuntime(self):
        self.timer.timeout.connect(self.next_frames)
        self.buttons["play"].clicked.connect(
            lambda x: self.change_status(not self.is_play))
        self.buttons["next"].clicked.connect(self.next_frames)
        self.buttons["prev"].clicked.connect(self.prev_frames)
        self.toggles["edges"].clicked.connect(self.toggle)
        self.toggles["cls"].clicked.connect(self.toggle)
        self.toggles["pre"].clicked.connect(self.toggle)
        self.playback.valueChanged.connect(self.traverse_frames)
        self.sli_span.valueChanged.connect(self.change_span)
        self.sli_thre.valueChanged.connect(self.change_thre)

    def toggle(self):
        if self.toggles["edges"].isChecked():
            self.imgs_show = self.app.IMGS["edg"]

        elif self.toggles["cls"].isChecked():
            self.imgs_show = self.app.OUTS["pred_cls"]

        elif self.toggles["pre"].isChecked():
            self.imgs_show = self.app.OUTS["pred"]

        self.update_frames()

    def initUI(self):
        # Slider
        self.lb_thre.setAlignment(Qt.AlignCenter)
        self.sli_thre.setMinimum(500)
        self.sli_thre.setMaximum(1000)
        self.sli_thre.setValue(1000)
        self.sli_thre.setTickPosition(QSlider.NoTicks)
        self.sli_thre.setTickInterval(1)
        self.sli_thre.setVisible(False)

        self.lb_span.setAlignment(Qt.AlignCenter)
        self.sli_span.setMinimum(1)
        self.sli_span.setMaximum(20)
        self.sli_span.setValue(1)
        self.sli_span.setTickPosition(QSlider.NoTicks)
        self.sli_span.setTickInterval(1)
        self.sli_span.setVisible(False)

        self.playback.setMinimum(0)
        self.playback.setMaximum(self.n_frame - 1)
        self.playback.setValue(0)
        self.playback.setTickPosition(QSlider.NoTicks)
        self.playback.setTickInterval(1)

        # Layout
        layout = QGridLayout(self)
        self.frame.setSizePolicy(QSizePolicy.Maximum,
                                 QSizePolicy.Maximum)
        self.frame.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # layout.addWidget(self.buttons["browse"], 0, 0, 1, 4)
        layout.addWidget(self.frame,             1, 0, 1, 3, alignment=Qt.AlignLeft)
        layout.addWidget(self.plot,              1, 3, 1, 1, alignment=Qt.AlignRight)
        layout.addWidget(self.lb_frame,          2, 0, 1, 1)
        layout.addWidget(self.playback,          3, 0, 1, 3)
        layout.addWidget(self.toggles["edges"],  4, 0, 1, 1)
        layout.addWidget(self.toggles["cls"],    4, 1, 1, 1)
        layout.addWidget(self.toggles["pre"],    4, 2, 1, 1)
        # layout.addWidget(self.sli_span,          5, 0, 1, 3)
        # layout.addWidget(self.lb_thre,           6, 0, 1, 3)
        # layout.addWidget(self.sli_thre,          7, 0, 1, 3)
        layout.addWidget(self.buttons["prev"],   8, 0)
        layout.addWidget(self.buttons["play"],   8, 1)
        layout.addWidget(self.buttons["next"],   8, 2)
        self.setLayout(layout)

        self.move(300, 200)
        self.setWindowTitle('Virtual Tags')
        self.setGeometry(50, 50, 1465, 620)

        self.show()

    def set_plot(self):
        # clear plot
        self.plot.clear()

        # obtain app info
        i   = self.i_frame
        pcs = self.app.OUTS["pcs"][i]
        ids = self.app.OUTS["k_to_id"][i]
        n   = len(pcs)

        # define position for dots
        pos = [(p[0], p[1]) for p in pcs]

        # define colors for dots
        if self.toggles["edges"].isChecked():
            bs = [QBrush(QColor("#000000")) for i in range(n)]

        elif self.toggles["cls"].isChecked():
            nc = len(colorsets) - 1  # exclude 0: background color
            bs = [QBrush(QColor(colorsets[(i % nc) + 1])) for i in range(n)]

        elif self.toggles["pre"].isChecked():
            bs = [QBrush(QColor(colorsets[idx.astype(int)])) for idx in ids]

        data = [dict(pos=pos[i], brush=bs[i], size=20) for i in range(n)]
        item_scatter = pg.ScatterPlotItem(data)
        self.plot.addItem(item_scatter)

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
        self.set_plot()
        self.playback.setValue(i)
        self.lb_frame.setText("Frame: %d" % i)
        self.frame.setPixmap(QPixmap(self.paths[i]))
        # self.frame.set_center(self.cx[i], self.cy[i])
        self.frame.set_predict(self.imgs_show[i])
        self.frame.repaint()

    def change_span(self):
        self.lb_span.setText("Span: %d" % self.sli_span.value())
        self.update_frames()

    def change_thre(self):
        self.lb_thre.setText("Threshold: %.3f" % (self.sli_thre.value()/1000))
        self.update_frames()

    def traverse_frames(self):
        self.i_frame = self.playback.value()
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


class QFrame(QLabel):
    '''
    Will keep imgRaw, imgVis and imgQmap
    '''

    def __init__(self):
        super().__init__()
        self.pixmap = None
        self.img_detect = None
        self.show_detect = True
        self.cx = -20
        self.cy = -20

    def set_image(self, pixmap):
        self.pixmap = pixmap

    def set_predict(self, img):
        self.img_detect = getIdx8QImg(img,  int(np.max(img)))

    def set_center(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.show_detect:
            painter.setOpacity(0.8)
            painter.drawPixmap(0, 0, self.img_detect)

            pen = QPen()
            pen.setWidth(8)
            pen.setStyle(Qt.SolidLine)
            pen.setColor(Qt.red)
            painter.setPen(pen)
            drawCross(self.cx, self.cy, painter, size=6)

        if self.pixmap is not None:
            painter.setOpacity(0.0)
            painter.drawPixmap(0, 0, self.pixmap)

        painter.end()


def getRGBQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w, h, w*3, QImage.Format_RGB888)
    return QPixmap(qImg)


def getBinQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Indexed8)
    qImg.setColor(0, qRgb(0, 0, 0))
    qImg.setColor(1, qRgb(241, 225, 29))
    return QPixmap(qImg)


def getIdx8QImg(img, k): # k=20
    colormap = [QColor(c) for c in colorsets]

    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Indexed8)
    nc = len(colormap) - 1  # exclude 0: background color

    # background color
    qImg.setColor(0, colormap[0].rgba())
    # cluster color
    for i in range(k): # i: 1 ~ 20
        qImg.setColor(i + 1, colormap[(i % nc) + 1].rgba()) # use '%' to iterate the colormap
    return QPixmap(qImg)


def getGrayQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Grayscale8)
    return QPixmap(qImg)



# Note
# self.thread = QThread(self)
# self.thread.started.connect(self.update_image)

    #     s = p1.size().expandedTo(p2.size())
    # result = QtGui.QPixmap(s)
    # result.fill(QtCore.Qt.transparent)
    # painter = QtGui.QPainter(result)
    # painter.setRenderHint(QtGui.QPainter.Antialiasing)
    # painter.drawPixmap(QtCore.QPoint(), p1)
    # painter.setCompositionMode(mode)
    # painter.drawPixmap(result.rect(), p2, p2.rect())
    # painter.end()
    # return result

# colormap = [qRgb(0, 0, 0),       # 0
#             qRgb(255, 255, 51),  # 1
#             qRgb(55, 126, 184),  # 2
#             qRgb(77, 175, 74),   # 3
#             qRgb(228, 26, 28),   # 4
#             qRgb(152, 78, 163),  # 5
#             qRgb(255, 127, 0),   # 6
#             qRgb(13, 136, 250),  # 7
#             qRgb(247, 129, 191),  # 8
#             qRgb(153, 153, 153)]  # 9
