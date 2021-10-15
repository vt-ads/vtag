from typing_extensions import runtime
from PyQt5.QtGui import QCursor
from pandas.core import frame
from lib import *
from VTags import VTags
from VTPlayback import VTPlayback
from DnD import DnDLineEdit
from VTFrame import VTFrame

class VTPlayer(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setMouseTracking(True)

        # wd
        self.wd       = ""
        self.dataname = ""
        self.args     = args
        # Predictions
        self.imgs_show = []
        self.alpha     = 200
        # Frames
        self.n_frame  = 0
        self.i_frame  = 0
        # Status
        self.is_play       = False
        self.is_press      = False
        self.has_anno      = False
        self.label_counter = 0
        # Setup timer
        self.timer  = QTimer(self)

        # init
        self.init_args()
        self.init_UI()
        self.init_runtime()
        self.init_data()
        self.setFocus(True)

    def init_data(self):
        self.paths = ls_files()
        try:
            # try loading VTag outputs
            self.load_annotations()
            self.playback.set_n(self.n_frame)
            self.has_anno          = True
            self.frame.show_detect = True
            self.groups["anno"].setEnabled(True)
        except Exception as e:
            self.n_frame = len(self.paths)
            self.playback.set_n(self.n_frame)
            self.playback.set_heat([0] * self.n_frame)
            self.has_anno = False
            self.frame.show_detect = False
            self.groups["anno"].setEnabled(False)
            print(e)
        self.update_frames()

    def init_args(self, # default arguments
                        dataname="group",
                        fps=10/1000):
        # extract args
        args = self.args
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
        self.frame     = VTFrame()
        self.playback  = VTPlayback()

        self.texts     = dict(wd       = DnDLineEdit())
        self.groups    = dict(display  = QGroupBox("Display Mode"),
                              anno     = QGroupBox("Annotations"))
        self.labels    = dict(frame    = QLabel("Frame: %d" % self.i_frame),
                              fps      = QLabel("Frame per second (FPS): %d" %
                                                int(self.fps * 1000)),
                              wd       = QLabel("Data directory"),
                              alpha    = QLabel("Opacity: %d / 255" % self.alpha),
                              describe = QLabel("Press Space to play/pause"))
        self.buttons   = dict(wd       = QPushButton("Browse"),
                              play     = QPushButton(""),
                              next     = QPushButton(""),
                              prev     = QPushButton(""),
                              run      = QPushButton("Analyze video"),
                              save     = QPushButton("Save labels"))
        self.check     = dict(lbs      = QCheckBox("Show labels"),
                              contours = QCheckBox("Show contours"))
        self.sliders   = dict(fps      = QSlider(Qt.Horizontal, self),
                              alpha    = QSlider(Qt.Horizontal, self))
        self.toggles   = dict(edges    = QRadioButton("Edges"),
                              cls      = QRadioButton("Clusters"),
                              pre      = QRadioButton("Predictions"))
        self.globalrec = dict(frame    = QRect(0, 0, 0, 0),
                              play     = QRect(0, 0, 0, 0))

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

    def load_annotations(self):
        self.ARGS, self.IMGS, self.OUTS = pickle.load(open("model.h5", "rb"))
        self.n_frame = self.ARGS["n"]
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
        self.wd = QFileDialog().getExistingDirectory(self, "", "")
        os.chdir(self.wd)
        self.dataname = self.wd.split("/")[-1]
        self.texts["wd"].setText(self.wd)
        self.labels["wd"].setText("Data directory: %s" % self.dataname)
        self.init_data()

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
        self.groups["display"] = QGroupBox("Display mode")
        layout_grp_display = QVBoxLayout()
        layout_grp_display.addWidget(self.toggles["edges"])
        layout_grp_display.addWidget(self.toggles["cls"])
        layout_grp_display.addWidget(self.toggles["pre"])
        self.groups["display"].setLayout(layout_grp_display)

        self.groups["anno"] = QGroupBox("Annotations")
        layout_grp_ann = QGridLayout(self)
        layout_grp_ann.addWidget(
            self.labels["alpha"], 0, 0, 1, 2, alignment=Qt.AlignBottom)
        layout_grp_ann.addWidget(
            self.sliders["alpha"], 1, 0, 1, 2, alignment=Qt.AlignTop)
        layout_grp_ann.addWidget(self.groups["display"], 2, 0, 2, 1)
        layout_grp_ann.addWidget(self.check["lbs"],      2, 1)
        layout_grp_ann.addWidget(self.check["contours"], 3, 1)
        self.groups["anno"].setLayout(layout_grp_ann)

        layout_config = QGridLayout(self)
        layout_config.addWidget(self.labels["wd"], 0, 0, alignment=Qt.AlignBottom)
        layout_config.addWidget(self.texts["wd"], 1, 0, 1, 3, alignment=Qt.AlignTop)
        layout_config.addWidget(self.buttons["wd"], 1, 3, alignment=Qt.AlignTop)
        layout_config.addWidget(self.labels["fps"], 2, 0, 1, 4, alignment=Qt.AlignBottom)
        layout_config.addWidget(self.sliders["fps"], 3, 0, 1, 4, alignment=Qt.AlignTop)
        layout_config.addWidget(self.groups["anno"], 4, 0, 1, 4)
        self.config.setLayout(layout_config)

        # layout main
        layout = QGridLayout(self)
        layout.addWidget(self.frame,  0, 0, 1, 4, alignment=Qt.AlignCenter)
        layout.addWidget(self.tabs,   0, 4, 1, 2, alignment=Qt.AlignCenter)
        layout.addWidget(self.buttons["prev"], 1, 0)
        layout.addWidget(self.buttons["play"], 1, 1)
        layout.addWidget(self.buttons["next"], 1, 2)
        layout.addWidget(self.playback,  1, 3)
        layout.addWidget(self.labels["frame"], 2, 0, 1, 3)
        layout.addWidget(self.labels["describe"], 2, 3)
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
        # when at the last frame
        if self.i_frame == self.n_frame:
            self.change_status(to_play=False)
            self.i_frame = 0
        self.playback.set_frame_tmp(self.i_frame)
        self.update_frames()

    def prev_frames(self):
        self.i_frame -= 1
        self.playback.set_frame_tmp(self.i_frame)
        self.update_frames()

    def update_frames(self):
        i = self.i_frame
        self.labels["frame"].setText("Frame: %d / %d" % (i, self.n_frame))
        self.frame.setPixmap(QPixmap(self.paths[i]))
        # set cursor in the playback
        self.playback.set_frame(self.i_frame)
        # if is playing, update tmp frame in the playback
        if self.is_play:
            self.playback.set_frame_tmp(self.i_frame)

        # show labels from the displayed frames
        if self.has_anno:
            self.frame.set_predict(self.imgs_show[i])
            self.set_plot()
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
        self.is_press = True
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
            self.playback.set_frame_tmp(self.i_frame)
            if self.is_play:
                x_mouse = evt.pos().x()
                self.i_frame = self.x_to_frame(x_mouse)
            self.update_frames()

    def mouseReleaseEvent(self, evt):
        self.is_press = False

    def mouseMoveEvent(self, evt):
        self.update_globalrec()
        if self.globalrec["play"].contains(evt.pos()):
            if (not self.is_play) or (self.is_play & self.is_press):
                x_mouse = evt.pos().x()
                self.i_frame = self.x_to_frame(x_mouse)
        else:
            self.i_frame = self.playback.i_frame_tmp
        self.update_frames()

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Space:
            self.change_status(not self.is_play)

    def x_to_frame(self, x):
        x_play = self.playback.mapToParent(QPoint(0, 0)).x()
        frame = int((x - x_play) // self.playback.bin)
        if frame > (self.n_frame - 1):
            frame = self.n_frame - 1
        return frame

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
