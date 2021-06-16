from lib import *
from Tags import VTags


class Player(QWidget):
    # def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data/one_pig"):
    def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data/group"):
    # def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data/group_small"):
    # def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data/one_pig_small"):
    # def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data/one_pig_all"):
        super().__init__()

        # WD
        os.chdir(folder)

        # Predictions
        self.img_bin = []
        # Frames
        self.folder  = folder
        self.paths   = ls_files(folder)
        self.frame   = QFrame()
        self.n_frame = len(self.paths)
        self.i_frame = 0
        self.lb_frame = QLabel("Frame: %d" % self.i_frame)
        self.fps     = 10 / 1000

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
        self.playback = QSlider(Qt.Horizontal, self)

        # init
        self.compute()
        self.update_frames()
        self.initRuntime()
        self.initUI()

    def compute(self):
        app = VTags(k=2)
        app.load(h5="out.h5")
        # print("--- load ---", flush=True)
        # app.load(self.folder)
        # print("--- move ---", flush=True)
        # app.detect_movements()
        # print("--- edge ---", flush=True)
        # app.detect_edges()
        # print("--- noise ---", flush=True)
        # app.remove_noise()
        # print("--- cluster ---", flush=True)
        # app.detect_clusters()
        # print("--- sort ---", flush=True)
        # app.sort_clusters()
        # print("--- k to id ---", flush=True)
        # app.map_k_to_id()
        # print("--- predict ---", flush=True)
        # app.make_predictions()
        self.img_bin = app.OUTS["pred"]

    def initRuntime(self):
        self.timer.timeout.connect(self.next_frames)
        self.buttons["play"].clicked.connect(
            lambda x: self.change_status(not self.is_play))
        self.buttons["next"].clicked.connect(self.next_frames)
        self.buttons["prev"].clicked.connect(self.prev_frames)
        self.playback.valueChanged.connect(self.traverse_frames)
        self.sli_span.valueChanged.connect(self.change_span)
        self.sli_thre.valueChanged.connect(self.change_thre)

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

        layout.addWidget(self.buttons["browse"], 0, 0, 1, 3)
        layout.addWidget(self.frame,             1, 0, 1, 3, alignment=Qt.AlignCenter)
        layout.addWidget(self.lb_frame,          2, 0, 1, 3)
        layout.addWidget(self.playback,          3, 0, 1, 3)
        # layout.addWidget(self.lb_span,           4, 0, 1, 3)
        # layout.addWidget(self.sli_span,          5, 0, 1, 3)
        # layout.addWidget(self.lb_thre,           6, 0, 1, 3)
        # layout.addWidget(self.sli_thre,          7, 0, 1, 3)
        layout.addWidget(self.buttons["prev"],   8, 0)
        layout.addWidget(self.buttons["play"],   8, 1)
        layout.addWidget(self.buttons["next"],   8, 2)
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
        self.playback.setValue(i)
        self.lb_frame.setText("Frame: %d" % i)
        self.frame.setPixmap(QPixmap(self.paths[i]))
        # self.frame.set_center(self.cx[i], self.cy[i])
        self.frame.set_predict(self.img_bin[i])
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
        self.show_detect = True
        self.img_detect = None
        self.cx = -20
        self.cy = -20

    def set_image(self, pixmap):
        self.pixmap = pixmap

    def set_predict(self, img):
        self.img_detect = getIdx8QImg(img, 10)

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
            print("sss")
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


def getIdx8QImg(img, k):
    colormap = [qRgb(0, 0, 0),
                qRgb(255, 255, 51),
                qRgb(55, 126, 184),
                qRgb(77, 175, 74),
                qRgb(228, 26, 28),
                qRgb(152, 78, 163),
                qRgb(255, 127, 0),
                qRgb(13, 136, 250),
                qRgb(247, 129, 191),
                qRgb(153, 153, 153)]
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Indexed8)
                #   h, w*3, QImage.Format_RGB888)
    for i in range(k):
        qImg.setColor(i, colormap[i])
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
