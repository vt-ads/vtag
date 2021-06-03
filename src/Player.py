from lib import *


class Player(QWidget):
    def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data/one_pig_all"):
        super().__init__()

        # WD
        os.chdir(folder)

        # Frames
        self.paths   = ls_files(folder)
        self.np_imgs = load_np(self.paths, n_imgs=300)
        self.frame   = QFrame()
        self.n_frame = len(self.paths)
        self.i_frame = 0
        self.fps     = 20 / 1000

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
        # Slider
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_frame - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.setTickInterval(1)

        # Layout
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

        # detect images
        # self.frame.show_detect = False
        img_detect = detect_imgs(self.np_imgs, i)
        self.frame.img_detect = getBinQImg(img_detect)
        self.frame.repaint()

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


class QFrame(QLabel):
    '''
    Will keep imgRaw, imgVis and imgQmap
    '''

    def __init__(self):
        super().__init__()
        # self.img_raw = img
        # self.img_vis = img[:, :, :3].copy()
        self.show_detect = True
        self.img_detect = None

        self.qimg = None
        self.isFitWidth = None
        self.rgX, self.rgY = (0, 0), (0, 0)
        self.sizeImg = (0, 0)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.show_detect:
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, self.img_detect)
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
    colormap = [qRgb(228, 26, 28),
                qRgb(55, 126, 184),
                qRgb(77, 175, 74),
                qRgb(152, 78, 163),
                qRgb(255, 127, 0),
                qRgb(255, 255, 51),
                qRgb(166, 86, 40),
                qRgb(247, 129, 191),
                qRgb(153, 153, 153)]
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Indexed8)
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
