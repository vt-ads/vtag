from typing_extensions import runtime
from PyQt5.QtGui import QCursor
from pandas.core import frame
from lib import *
from Tags import VTags
from Playback import Playback
from DnD import DnDLineEdit

class Player(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setMouseTracking(True)

        # wd
        self.wd = ""
        self.dataname = ""
        # Predictions
        self.imgs_show = []
        self.alpha = 200
        # Frames
        self.n_frame  = 0
        self.i_frame  = 0
        # Status
        self.is_play       = False
        self.has_anno      = False
        self.label_counter = 0
        # Setup timer
        self.timer  = QTimer(self)

        # init
        self.init_args(args)
        self.init_UI()
        self.load_frames()
        self.init_runtime()
        try:
            # try loading VTag outputs
            self.load_annotations()
            self.has_anno = True
        except Exception as e:
            print(e)
        self.update_frames()

    def init_args(self, args,
                    # default arguments
                    dataname="group",
                    fps=10/1000):
        # extract args
        n_args = len(args)
        for i in range(n_args):
            if args[i]   == "-data":
                dataname = args[i + 1]
            elif args[i] == "-fps":
                fps = int(args[i + 1]) / 1000

        # load args
        self.fps      = fps
        self.dataname = dataname

        # change WD to data directory
        file_main = args[0]
        ls_path_to_main = file_main.split("/")[:-2]
        os.chdir(os.path.join(*ls_path_to_main, "data", dataname))
        self.wd = os.getcwd()

    def init_UI(self):
        # init components
        self.frame     = QFrame()
        self.playback  = Playback()

        self.texts     = dict(wd        = DnDLineEdit())
        self.labels    = dict(frame     = QLabel("Frame: %d" % self.i_frame),
                              fps       = QLabel("Frame per second (FPS): %d" %
                                                 int(self.fps * 1000)),
                              wd        = QLabel("Data directory"),
                              alpha     = QLabel("Opacity: %d / 255" % self.alpha))
        self.buttons   = dict(wd        = QPushButton("Browse"),
                              play      = QPushButton(""),
                              next      = QPushButton(""),
                              prev      = QPushButton(""),
                              run       = QPushButton("Analyze video"),
                              save      = QPushButton("Save labels"))
        self.check     = dict(lbs       = QCheckBox("Show labels"),
                              contours  = QCheckBox("Show contours"))
        self.sliders   = dict(fps       = QSlider(Qt.Horizontal, self),
                              alpha     = QSlider(Qt.Horizontal, self))
        self.toggles   = dict(edges     = QRadioButton("Edges"),
                              cls       = QRadioButton("Clusters"),
                              pre       = QRadioButton("Predictions"))
        self.globalrec = dict(frame     = QRect(0, 0, 0, 0),
                              play      = QRect(0, 0, 0, 0))

        # WD
        self.texts["wd"].setText(self.wd)
        self.labels["wd"].setText("Data directory: %s" % self.dataname)

        # set icons
        # https://joekuan.files.wordpress.com/2015/09/screen3.png
        self.buttons["play"].setIcon(
            self.style().standardIcon(getattr(QStyle, "SP_MediaPlay")))
        self.buttons["next"].setIcon(
            self.style().standardIcon(getattr(QStyle, "SP_MediaSeekForward")))
        self.buttons["prev"].setIcon(
            self.style().standardIcon(getattr(QStyle, "SP_MediaSeekBackward")))
        self.buttons["wd"].setIcon(
            self.style().standardIcon(getattr(QStyle, "SP_DialogOpenButton")))

        # checkboxes
        self.check["lbs"].setChecked(True)
        self.check["contours"].setChecked(True)

        # tabs
        self.config = QWidget()
        self.plot   = pg.plot()
        self.tabs   = QTabWidget()
        self.tabs.addTab(self.config, "Configuration")
        self.tabs.addTab(self.plot,   "PCA")

        # sliders
        self.sliders["fps"].setMinimum(1)
        self.sliders["fps"].setMaximum(60)
        self.sliders["fps"].setValue(int(self.fps * 1000))
        self.sliders["fps"].setTickPosition(QSlider.NoTicks)
        self.sliders["fps"].setTickInterval(1)

        self.sliders["alpha"].setMinimum(1)
        self.sliders["alpha"].setMaximum(255)
        self.sliders["alpha"].setValue(self.alpha)
        self.sliders["alpha"].setTickPosition(QSlider.NoTicks)
        self.sliders["alpha"].setTickInterval(1)

        # finalize
        self.set_layout()
        self.move(300, 200)
        self.setWindowTitle('Virtual Tags')
        self.setGeometry(50, 50, 1400, 550)
        self.show()

    def load_frames(self):
        self.paths   = ls_files()
        self.n_frame = len(self.paths)
        self.playback.set_n(self.n_frame)

    def load_annotations(self):
        self.ARGS, self.IMGS, self.OUTS = pickle.load(open("model.h5", "rb"))
        # define predictions
        self.toggles["pre"].setChecked(True)
        self.imgs_show = self.IMGS["pred"]  # define what show on the screen
        # load heat map
        if self.ARGS["n_id"] == 2:
            pre_grp = np.array(pd.read_csv("labels.csv")
                                ).reshape((self.n_frame, 2, 2))
            dist = np.array([distance(p1, p2) for p1, p2 in pre_grp])
            self.playback.set_heat(dist)
        # try loading existing labels
        labels = pd.read_csv("labels.csv")
        self.OUTS["pred_labels"] = lb_from_pd_to_np(labels)

    def init_runtime(self):
        self.timer.timeout.connect(self.next_frames)
        self.check["lbs"].stateChanged.connect(self.check_lbs)
        self.check["contours"].stateChanged.connect(self.check_contours)
        self.buttons["wd"].clicked.connect(self.browse_wd)
        self.buttons["play"].clicked.connect(
            lambda x: self.change_status(not self.is_play))
        self.buttons["next"].clicked.connect(self.next_frames)
        self.buttons["prev"].clicked.connect(self.prev_frames)
        self.buttons["save"].clicked.connect(self.save_lbs)
        self.toggles["edges"].clicked.connect(self.toggle)
        self.toggles["cls"].clicked.connect(self.toggle)
        self.toggles["pre"].clicked.connect(self.toggle)
        self.sliders["fps"].valueChanged.connect(self.set_fps)
        self.sliders["alpha"].valueChanged.connect(self.set_alpha)

    def browse_wd(self):
        # fileter = "Images (*.tif *.jpg *.jpeg *.png)"
        # path = QFileDialog().getExistingDirectory(self, "", "", fileter)[0]
        self.wd = QFileDialog().getExistingDirectory(self, "", "")
        self.dataname = self.wd.split("/")[-1]
        self.texts["wd"].setText(self.wd)
        self.labels["wd"].setText("Data directory: %s" % self.dataname)

    def set_fps(self):
        new_fps = self.sliders["fps"].value()
        self.labels["fps"].setText("Frame per second (FPS): %d" % (new_fps))
        self.fps = new_fps / 1000
        self.change_status(True)
        self.update_frames()

    def set_alpha(self):
        alpha = self.sliders["alpha"].value()
        self.labels["alpha"].setText("Opacity: %d / 255" % alpha)
        self.frame.alpha = alpha
        self.update_frames()

    def check_lbs(self):
        self.frame.show_lbs = self.check["lbs"].isChecked()
        self.update_frames()

    def check_contours(self):
        self.frame.show_detect = self.check["contours"].isChecked()
        self.update_frames()

    def toggle(self):
        if self.toggles["edges"].isChecked():
            self.imgs_show = self.IMGS["edg"]

        elif self.toggles["cls"].isChecked():
            self.imgs_show = self.IMGS["pred_cls"]

        elif self.toggles["pre"].isChecked():
            self.imgs_show = self.IMGS["pred"]

        self.update_frames()

    def save_lbs(self):
        labels = self.OUTS["pred_labels"]
        n_ids  = self.ARGS["n_id"]
        save_labels(labels, n_ids, "labels.csv")


    def set_layout(self):
        # style
        self.setStyleSheet("""
        QWidget {
            font: 16pt Trebuchet MS
        }
        QGroupBox::title{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QGroupBox {
            border: 1px solid gray;
            border-radius: 9px;
            margin-top: 0.5em;
        }
        """)

        # layout tab
        layout_tab = QGridLayout(self)
        layout_tab.addWidget(self.labels["wd"], 0, 0, alignment=Qt.AlignBottom)
        layout_tab.addWidget(self.texts["wd"], 1, 0, 1, 3, alignment=Qt.AlignTop)
        layout_tab.addWidget(self.buttons["wd"], 1, 3, alignment=Qt.AlignTop)
        layout_tab.addWidget(self.labels["fps"], 2, 0, 1, 2, alignment=Qt.AlignBottom)
        layout_tab.addWidget(self.sliders["fps"], 3, 0, 1, 2, alignment=Qt.AlignTop)
        layout_tab.addWidget(self.labels["alpha"], 2, 2, 1, 2, alignment=Qt.AlignBottom)
        layout_tab.addWidget(self.sliders["alpha"], 3, 2, 1, 2, alignment=Qt.AlignTop)

        grp_display = QGroupBox("Display mode")
        layout_grp_display = QVBoxLayout()
        layout_grp_display.addWidget(self.toggles["edges"])
        layout_grp_display.addWidget(self.toggles["cls"])
        layout_grp_display.addWidget(self.toggles["pre"])
        grp_display.setLayout(layout_grp_display)

        grp_ann = QGroupBox("Annotations")
        layout_grp_ann = QVBoxLayout()
        layout_grp_ann.addWidget(self.check["lbs"])
        layout_grp_ann.addWidget(self.check["contours"])
        grp_ann.setLayout(layout_grp_ann)

        layout_tab.addWidget(grp_display,  4, 0, 1, 2)
        layout_tab.addWidget(grp_ann,      4, 2, 1, 2)
        self.config.setLayout(layout_tab)

        # layout main
        layout = QGridLayout(self)
        layout.addWidget(self.frame,     0, 0, 1, 4, alignment=Qt.AlignCenter)
        layout.addWidget(self.tabs,      0, 4, 1, 2, alignment=Qt.AlignCenter)
        layout.addWidget(self.labels["frame"], 1, 0, 1, 3)
        layout.addWidget(self.buttons["prev"], 2, 0)
        layout.addWidget(self.buttons["play"], 2, 1)
        layout.addWidget(self.buttons["next"], 2, 2)
        layout.addWidget(self.playback,        1, 3, 2, 1)
        layout.addWidget(self.buttons["run"],  1, 4, 2, 1)
        layout.addWidget(self.buttons["save"], 1, 5, 2, 1)
        self.setLayout(layout)

        # align & size
        self.tabs.setSizePolicy(QSizePolicy.Expanding,
                                QSizePolicy.Expanding)
        self.buttons["save"].setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)
        self.buttons["run"].setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Expanding)
        self.frame.setSizePolicy(QSizePolicy.Expanding,
                                 QSizePolicy.Expanding)
        self.playback.setSizePolicy(QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)


    def set_plot(self):
        # clear plot
        self.plot.clear()

        # obtain app info
        i   = self.i_frame
        pcs = self.OUTS["pcs"][i]
        ids = self.OUTS["k_to_id"][i]
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
        self.labels["frame"].setText("Frame: %d" % i)
        self.frame.setPixmap(QPixmap(self.paths[i]))
        self.frame.set_predict(self.imgs_show[i])
        self.playback.set_frame(self.i_frame)
        # show labels from the displayed frames
        k      = self.ARGS["n_id"]
        labels = self.OUTS["pred_labels"]
        x      = labels[i, :, 1]
        y      = labels[i, :, 0]
        self.frame.set_label(x, y)
        # update GUI
        self.frame.repaint()
        self.update_globalrec()

    def update_globalrec(self):
        self.globalrec["frame"] = QRect(self.frame.mapToParent(QPoint(0, 0)),
                                        self.frame.size())
        self.globalrec["play"] = QRect(self.playback.mapToParent(QPoint(0, 0)),
                                        self.playback.size())

    def traverse_frames(self):
        self.i_frame = self.playback.value()
        self.update_frames()

    def change_status(self, to_play):
        if to_play:
            self.is_play = True
            self.buttons["play"].setIcon(
                self.style().standardIcon(getattr(QStyle, "SP_MediaPause")))
            self.timer.start(int(1 / self.fps))

        else:
            self.is_play = False
            self.buttons["play"].setIcon(
                self.style().standardIcon(getattr(QStyle, "SP_MediaPlay")))
            self.timer.stop()

    def mousePressEvent(self, evt):
        self.update_globalrec()
        if self.globalrec["frame"].contains(evt.pos()):
            # collect info
            k       = self.ARGS["n_id"]
            labels  = self.OUTS["pred_labels"]
            counter = self.label_counter
            i       = self.i_frame

            # update label counter
            self.label_counter = (counter + 1) % k
            self.frame.label_counter = self.label_counter

            # get labels
            x, y = self.frame.mx, self.frame.my

            # enter labels
            labels[i, counter, 1] = x
            labels[i, counter, 0] = y

            # if label all ids, move to next frame
            if ((counter + 1) % k) == 0:
                self.next_frames()
            else:
                self.update_frames()

        elif self.globalrec["play"].contains(evt.pos()):
            self.change_status(not self.is_play)

    def mouseMoveEvent(self, evt):
        self.update_globalrec()
        if self.globalrec["play"].contains(evt.pos()):
            x_mouse = evt.pos().x()
            x_play  = self.playback.mapToParent(QPoint(0, 0)).x()
            frame   = int((x_mouse - x_play) // self.playback.bin)
            if frame > (self.n_frame - 1):
                frame = self.n_frame - 1
            self.i_frame = frame
        self.update_frames()


class QFrame(QLabel):
    '''
    Will keep imgRaw, imgVis and imgQmap
    '''

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.pixmap      = None
        self.img_detect  = None
        self.show_detect = True
        self.cx = -20
        self.cy = -20
        # mouse moving events
        self.mx = -1
        self.my = -1
        # ground truth
        self.alpha    = 200
        self.show_lbs = True
        self.lb_x = []
        self.lb_y = []
        # label counter
        self.label_counter = 0

    def set_image(self, pixmap):
        self.pixmap = pixmap

    def set_predict(self, img):
        self.img_detect = getIdx8QImg(img,  int(np.max(img)), alpha=self.alpha)

    def set_center(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def set_label(self, lb_x, lb_y):
        self.lb_x, self.lb_y = lb_x, lb_y

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.show_detect:
            # draw detection
            painter.drawPixmap(0, 0, self.img_detect)

        # draw labels
        if self.show_lbs:
            n_labels = len(self.lb_x)

            pen = QPen()
            pen.setStyle(Qt.SolidLine)
            for i in range(n_labels):
                # border (black)
                pen.setWidth(10)
                pen.setColor(QColor(colorsets[0]))
                painter.setPen(pen)
                drawCross(self.lb_x[i], self.lb_y[i], painter, size=6)
                # filled (color)
                pen.setWidth(8)
                pen.setColor(QColor(colorsets[i + 1]))
                painter.setPen(pen)
                drawCross(self.lb_x[i], self.lb_y[i], painter, size=6)

        # if self.pixmap is not None:
        #     print("pixmap")
        #     painter.setOpacity(0.0)
        #     painter.drawPixmap(0, 0, self.pixmap)

        painter.end()

         # cursor
        img_cur = QPixmap(30, 30)
        img_cur.fill(QColor(0, 0, 0, 0))
        paint_cur = QPainter(img_cur)
        paint_cur.drawPixmap(0, 0, img_cur)
        pen = QPen()
        pen.setStyle(Qt.SolidLine)
        i = self.label_counter
        # border (black)
        pen.setWidth(10)
        pen.setColor(QColor(colorsets[0]))
        paint_cur.setPen(pen)
        drawCross(15, 15, paint_cur, size=6)
        # filled (color)
        pen.setWidth(8)
        pen.setColor(QColor(colorsets[i + 1]))
        paint_cur.setPen(pen)
        drawCross(15, 15, paint_cur, size=6)
        # set curor
        self.setCursor(QCursor(img_cur))
        paint_cur.end()

    def mouseMoveEvent(self, evt):
        self.mx, self.my = evt.x(), evt.y()

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


def getIdx8QImg(img, k, alpha=200): # k=20
    colormap = [QColor(c) for c in colorsets]

    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Indexed8)
    nc = len(colormap) - 1  # exclude 0: background color

    # background color
    # qImg.setColor(0, colormap[0].rgba())
    qImg.setColor(0, QColor(0, 0, 0, alpha).rgba())
    # cluster color
    for i in range(k):
        qImg.setColor(i + 1, colormap[(i % nc) + 1].rgba()) # use '%' to iterate the colormap
    return QPixmap(qImg)


def getGrayQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Grayscale8)
    return QPixmap(qImg)


colorsets = np.array(["#000000",
                      "#ffff33", "#f94144", "#f3722c", "#f8961e", "#f9844a",
                      "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590",
                      "#277da1"])

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
