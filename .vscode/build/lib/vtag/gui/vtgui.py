import os
from threading import local
import time

from PyQt6.QtCore    import Qt, QTimer, QPoint, QRect
from PyQt6.QtWidgets import QWidget, QGroupBox, QLabel,\
                            QPushButton, QRadioButton, QCheckBox,\
                            QSlider, QStyle, QTabWidget,\
                            QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy,\
                            QFileDialog, QMessageBox, QComboBox
from PyQt6.QtGui     import QPixmap

# vtag imports
from ..core.vtag import VTag
from .vtframe    import *
from .vtplayback import *
from .colors import vtcolor

class VTGUI(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        # Frames
        self.n_frame  = 0
        self.i_frame  = 0
        # Status
        self.is_play       = False
        self.is_press      = False
        self.is_load       = False
        # Display
        self.alpha = 50
        self.fps   = 10
        # --- k tag
        self.k     = 3
        self.size  = 70 # detect size
        self.max_k = 9
        self.i_tag = 0
        # Setup timer
        self.timer  = QTimer(self)
        # init components
        self.frame     = VTFrame()
        self.playback  = VTPlayback()
        self.vtag      = None

        # init
        self.init_UI()
        self.init_runtime()
        self.setFocus()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # TODO
        # self.load()
        # TODO

    def init_UI(self):
        self.panel  = dict(left     = QWidget(),
                           right    = QWidget(),
                           playback = QWidget(),
                           k        = QWidget())
        self.groups = dict(anno     = QGroupBox("Annotation"),
                           display  = QGroupBox("Display"))
        self.buttons = dict(load  = QPushButton("Load data"),
                            poi   = QPushButton("Estimate POI"),
                            track = QPushButton("Track"),
                            save_poi = QPushButton("Save POI"),
                            save  = QPushButton("Save"),
                            # media
                            play    = QPushButton(""),
                            next    = QPushButton(""),
                            prev    = QPushButton(""))
        self.labels    = dict(k     = QLabel("Tags: %d" % self.k),
                              track = QLabel("Tracker"),
                              size  = QLabel("Detect size: %d" % self.size),
                              frame = QLabel("Frame: %d" % self.i_frame),
                              fps   = QLabel("Frame per second (FPS): %d" %
                                                 int(self.fps)),
                              alpha    = QLabel("Opacity: %d / 255" % self.alpha),
                              describe = QLabel("Press 'Space' to play/pause. 'Arrow right/left' to the next/previous frame." ))
        self.check     = dict(lbs   = QCheckBox("Show labels"),
                              poi   = QCheckBox("Show POI"))
        self.sliders   = dict(k     = QSlider(Qt.Orientation.Horizontal, self),
                              size  = QSlider(Qt.Orientation.Horizontal, self),
                              fps   = QSlider(Qt.Orientation.Horizontal, self),
                              alpha = QSlider(Qt.Orientation.Horizontal, self))
        self.globalrec = dict(frame = QRect(0, 0, 0, 0),
                              play  = QRect(0, 0, 0, 0))
        self.combos    = dict(track = QComboBox())

        # for k tags
        self.tags = dict(toggle = [], color = [])

        # set icons
        # https://joekuan.files.wordpress.com/2015/09/screen3.png
        self.buttons["play"].setIcon(
            self.style().standardIcon(getattr(QStyle.StandardPixmap,
                                              "SP_MediaPlay")))
        self.buttons["next"].setIcon(
            self.style().standardIcon(getattr(QStyle.StandardPixmap,
                                              "SP_MediaSeekForward")))
        self.buttons["prev"].setIcon(
            self.style().standardIcon(getattr(QStyle.StandardPixmap,
                                              "SP_MediaSeekBackward")))
        self.buttons["load"].setIcon(
            self.style().standardIcon(getattr(QStyle.StandardPixmap,
                                              "SP_DialogOpenButton")))
        # checkboxes
        self.check["lbs"].setChecked(True)
        self.check["poi"].setChecked(False)

        # sliders
        self.sliders["k"].setMinimum(1)
        self.sliders["k"].setMaximum(self.max_k)
        self.sliders["k"].setValue(self.k)
        self.sliders["k"].setTickPosition(QSlider.TickPosition.NoTicks)
        self.sliders["k"].setTickInterval(1)

        self.sliders["size"].setMinimum(1)
        self.sliders["size"].setMaximum(300)
        self.sliders["size"].setValue(self.size)
        self.sliders["size"].setTickPosition(QSlider.TickPosition.NoTicks)
        self.sliders["size"].setTickInterval(1)

        self.sliders["fps"].setMinimum(1)
        self.sliders["fps"].setMaximum(60)
        self.sliders["fps"].setValue(int(self.fps))
        self.sliders["fps"].setTickPosition(QSlider.TickPosition.NoTicks)
        self.sliders["fps"].setTickInterval(1)

        self.sliders["alpha"].setMinimum(1)
        self.sliders["alpha"].setMaximum(255)
        self.sliders["alpha"].setValue(self.alpha)
        self.sliders["alpha"].setTickPosition(QSlider.TickPosition.NoTicks)
        self.sliders["alpha"].setTickInterval(1)

        # combo tracker
        self.combos["track"].addItem("SparseLK")
        self.combos["track"].addItem("CSRT")
        self.combos["track"].addItem("MIL")

        # finalize
        self.set_layout()
        self.setWindowTitle('VTag')
        # self.setGeometry(50, 50, 1400, 550)
        self.show()

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
        #---small components
        layout_anno = QVBoxLayout()
        layout_anno.addWidget(self.labels["k"])
        layout_anno.addWidget(self.sliders["k"])
        layout_anno.addWidget(self.panel["k"])
        layout_anno.addWidget(self.labels["size"])
        layout_anno.addWidget(self.sliders["size"])
        layout_anno.addWidget(self.labels["track"])
        layout_anno.addWidget(self.combos["track"])
        layout_anno.addWidget(self.buttons["track"])
        layout_anno.addWidget(self.buttons["poi"])
        self.groups["anno"].setLayout(layout_anno)
        self.set_layout_k()
        self.update_layout_k()

        layout_display = QGridLayout(self)
        layout_display.addWidget(self.labels["fps"],    0, 0, 1, 2)
        layout_display.addWidget(self.sliders["fps"],   1, 0, 1, 2)
        layout_display.addWidget(self.labels["alpha"],  2, 0, 1, 2)
        layout_display.addWidget(self.sliders["alpha"], 3, 0, 1, 2)
        layout_display.addWidget(self.check["lbs"],     4, 0, 1, 1)
        layout_display.addWidget(self.check["poi"],     4, 1, 1, 1)
        self.groups["display"].setLayout(layout_display)

        layout_playback = QGridLayout(self)
        layout_playback.addWidget(self.buttons["prev"],    0, 0)
        layout_playback.addWidget(self.buttons["play"],    0, 1)
        layout_playback.addWidget(self.buttons["next"],    0, 2)
        layout_playback.addWidget(self.labels["frame"],    1, 0, 1, 3)
        layout_playback.addWidget(self.labels["describe"], 0, 3, 2, 3)
        self.panel["playback"].setLayout(layout_playback)

        #---panels
        layout_left = QVBoxLayout()
        layout_left.addWidget(self.frame)
        layout_left.addWidget(self.playback)
        layout_left.addWidget(self.panel["playback"])
        self.panel["left"].setLayout(layout_left)

        layout_right = QVBoxLayout()
        # layout_right.addWidget(self.buttons["load"])
        layout_right.addWidget(self.groups["anno"])
        layout_right.addWidget(self.groups["display"])
        layout_right.addWidget(self.buttons["save_poi"])
        layout_right.addWidget(self.buttons["save"])
        self.panel["right"].setLayout(layout_right)

        # main
        layout = QHBoxLayout()
        layout.addWidget(self.panel["left"])
        layout.addWidget(self.panel["right"])
        self.setLayout(layout)

        # align & size
        self.frame.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        self.playback.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)

    def init_runtime(self):
        self.panel["left"].setMouseTracking(True)
        self.buttons["load"].clicked.connect(self.load)
        self.buttons["save"].clicked.connect(self.save)
        self.buttons["save_poi"].clicked.connect(self.save_poi)
        self.buttons["track"].clicked.connect(self.track)
        self.buttons["poi"].clicked.connect(self.estimate_poi)
        self.timer.timeout.connect(self.next_frames)
        self.check["lbs"].stateChanged.connect(self.update_frames)
        self.check["poi"].stateChanged.connect(self.update_frames)
        self.buttons["play"].clicked.connect(
            lambda x: self.change_status(not self.is_play))
        self.buttons["next"].clicked.connect(self.next_frames)
        self.buttons["prev"].clicked.connect(self.prev_frames)
        self.sliders["fps"].valueChanged.connect(self.set_fps)
        self.sliders["alpha"].valueChanged.connect(self.set_alpha)
        self.sliders["k"].valueChanged.connect(self.set_k)
        self.sliders["size"].valueChanged.connect(self.set_size)

    def update_layout_k(self):
        layout = self.panel["k"].layout()
        # remove existing widgets from the layout
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        # add needed widgets
        for i in range(self.k):
            self.tags["toggle"][i].setChecked(False)
            layout.addWidget(self.tags["color"][i],
                             0, i, Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.tags["toggle"][i],
                             1, i, Qt.AlignmentFlag.AlignCenter)
        # reset assigned i
        self.set_i_tag(0)

    def set_layout_k(self):
        self.panel["k"].setLayout(QGridLayout(self))
        for i in range(self.max_k):
            # toggle
            toggle = QRadioButton()
            self.tags["toggle"] += [toggle]
            # label
            label  = QLabel()
            colorbox = QPixmap(20, 20)
            colorbox.fill(vtcolor(i + 1))
            label.setPixmap(colorbox)
            self.tags["color"]  += [label]
            # runtime
            toggle.toggled.connect(self.toggle_tag)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    def load(self, path=False):
        # TESTING --------------------
        # TODO
        # wd = "/Users/niche/OneDrive - Virginia Tech/GitHub/vtag/data/pig_3_sample"
        # TESTING --------------------

        # try:
        if path is False:
            qdialog = QFileDialog()
            path, _ = qdialog.getOpenFileName(self, "", "")
        self.vtag = VTag()
        self.vtag.load(path=path)
        # get vtag arguments
        n = self.vtag.ARGS["n"]
        k = self.vtag.ARGS["k"]
        # update GUI
        self.set_n(n)
        self.sliders["k"].setValue(k) # to trigger set_k()
        self.is_load = True
        self.playback.set_error(self.vtag.DATA["error"])
        QMessageBox().information(self, "", "Data loaded successfully")
        # except Exception as e:
        #     # no png found, or not a valid path
        #     print(e)
        #     QMessageBox().information(self,
        #         "Failed to load files",
        #         "No valid data (.mp4, .png, .jpg) is found")

        # refresh screen
        self.update_frames()

    def toggle_tag(self):
        i = 0
        for i in range(self.k):
            if self.tags["toggle"][i].isChecked():
                break
        self.i_tag       = i
        self.frame.i_tag = i
        self.frame.repaint()

    def save_poi(self):
        try:
            self.vtag.save_mask(frame=self.i_frame)
            path = os.path.join(self.vtag.stream.dirsave, "mask")
            QMessageBox().information(self,
                "Annotation Saved",
                "The annotation has been saved to '%s'" % path)
        except:
            QMessageBox().information(self,
                    "Failed",
                    "No vtag object is found")

    def save(self):
        filename = "vtag_" + localtime() + ".h5"
        try:
            self.vtag.save(filename=filename)
            QMessageBox().information(self,
                "Annotation Saved",
                "The annotation has been saved to '%s'" % filename)
        except:
            QMessageBox().information(self,
                    "Failed",
                    "No vtag object is found")

    def track(self):
        if self.is_load:
            self.vtag.track(frame=self.i_frame,
                            tracker=self.combos["track"].currentText(),
                            pts_init=self.vtag.lbs(self.i_frame),
                            win_xy=(self.size, self.size))
            self.playback.set_error(self.vtag.DATA["error"])

        QMessageBox().information(self,
                "Track",
                "Tracking completed")

        self.update_frames()

    def estimate_poi(self):
        if self.is_load:
            self.vtag.detect_poi(self.i_frame)

        QMessageBox().information(self,
                "POI",
                "POI estimation completed")

        self.update_frames()

    # --- setter
    def set_i_tag(self, i):
        i = i % self.k
        self.tags["toggle"][i].setChecked(True)

    def set_n(self, n):
        self.n_frame = n
        self.i_frame = 0
        self.playback.set_n(n)
        self.update_frames()

    def set_size(self):
        self.size = self.sliders["size"].value()
        self.labels["size"].setText("Detect size: %d" % self.size)
        self.update_frames()

    def set_k(self):
        """
        Trigger by the GUI slider

        self.set_k()
            -> self.update_layout_k()
            -> self.set_i_tag()
            -> self.toggle_tag()
        """
        old_k = self.k
        new_k = self.sliders["k"].value()
        # change k
        self.k = new_k
        # update vtag
        if self.is_load:
            self.vtag.ARGS["k"] = new_k
            self.vtag.DATA["lbs"]   = reshape_matrix(
                                        mat_old=self.vtag.DATA["lbs"],
                                        shp_new=(self.n_frame, new_k, 2),
                                        dim_old=old_k,
                                        dim_new=new_k)
            self.vtag.DATA["area"]  = reshape_matrix(
                                        mat_old=self.vtag.DATA["area"],
                                        shp_new=(self.n_frame, new_k),
                                        dim_old=old_k,
                                        dim_new=new_k)
            self.vtag.DATA["error"] = reshape_matrix(
                                        mat_old=self.vtag.DATA["error"],
                                        shp_new=(self.n_frame, new_k),
                                        dim_old=old_k,
                                        dim_new=new_k)
        # update label
        self.labels["k"].setText("Tags: %d" % self.k)
        # update tag layout
        self.update_layout_k()

    def set_fps(self):
        self.fps = self.sliders["fps"].value()
        self.labels["fps"].setText("Frame per second (FPS): %d" % self.fps)
        self.change_status(True)
        self.update_frames()

    def set_alpha(self):
        alpha = self.sliders["alpha"].value()
        self.labels["alpha"].setText("Opacity: %d / 255" % alpha)
        self.frame.set_alpha(alpha)
        self.update_frames()

    def toggle(self):
        if self.toggles["edges"].isChecked():
            self.imgs_show = self.IMGS["edg"]

        elif self.toggles["cls"].isChecked():
            self.imgs_show = self.IMGS["pred_cls"]

        elif self.toggles["pre"].isChecked():
            self.imgs_show = self.IMGS["pred"]

        self.update_frames()

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

        if self.is_load:
            self.frame.set_image(self.vtag.img(i))
            self.playback.set_frame(self.i_frame)

            # if is playing, update tmp frame in the playback
            if self.is_play:
                self.playback.set_frame_tmp(self.i_frame)

            # update label
            if self.vtag.DATA["lbs"] is not None and self.check["lbs"].isChecked():
                self.frame.set_labels(self.vtag.lbs(i))
                self.frame.show_lbs = True
            else:
                self.frame.show_lbs = False
            # update poi
            if self.vtag.DATA["poi"] is not None and self.check["poi"].isChecked():
                self.frame.set_poi(self.vtag.mask(i))
                self.frame.show_poi = True
            else:
                self.frame.show_poi = False

        # update cursor in vtframe
        self.frame.size_m = self.size

        # update GUI
        self.frame.repaint()
        self.update_globalrec()

    def update_globalrec(self):
        self.globalrec["frame"] = QRect(self.frame.mapTo(self, QPoint(0, 0)),
                                        self.frame.size())
        self.globalrec["play"] = QRect(self.playback.mapTo(self, QPoint(0, 0)),
                                        self.playback.size())

    def change_status(self, to_play):
        if to_play:
            self.is_play = True
            self.buttons["play"].setIcon(
                self.style().standardIcon(getattr(QStyle.StandardPixmap,
                                                  "SP_MediaPause")))
            self.timer.start(int(1 / self.fps * 1000))

        else:
            self.is_play = False
            self.buttons["play"].setIcon(
                self.style().standardIcon(getattr(QStyle.StandardPixmap,
                                                  "SP_MediaPlay")))
            self.timer.stop()

    def mousePressEvent(self, evt):
        self.is_press = True
        self.update_globalrec()
        if self.is_load:
            if self.globalrec["frame"].contains(evt.position().toPoint()) and\
                evt.button() == Qt.MouseButton.LeftButton:
                # collect info
                k     = self.vtag.ARGS["k"]
                lbs   = self.vtag.DATA["lbs"]
                i_tag = self.i_tag
                i     = self.i_frame

                # update labels
                lbs[i, i_tag, 0] = self.frame.mx
                lbs[i, i_tag, 1] = self.frame.my

                # update label counter
                self.set_i_tag(i_tag + 1)

                # if label all ids, move to next frame
                if self.i_tag == 0:
                    self.next_frames()
                else:
                    self.update_frames()

            elif self.globalrec["play"].contains(evt.position().toPoint()):
                # determine which frame to traverse to in the playback bar
                self.playback.set_frame_tmp(self.i_frame)
                if self.is_play:
                    x_mouse = evt.position().x()
                    self.i_frame = self.x_to_frame(x_mouse)
                self.update_frames()

        # switch i_tag
        if evt.button() == Qt.MouseButton.RightButton:
            self.set_i_tag(self.i_tag + 1)

    def mouseReleaseEvent(self, evt):
        self.is_press = False

    def mouseMoveEvent(self, evt):
        self.update_globalrec()
        if self.globalrec["play"].contains(evt.position().toPoint()):
            if (not self.is_play) or (self.is_play & self.is_press):
                x_mouse = evt.position().x()
                self.i_frame = self.x_to_frame(x_mouse)
        else:
            self.i_frame = self.playback.i_frame_tmp
        self.update_frames()

    def wheelEvent(self, evt):
        # scroll down => positive
        delta = evt.pixelDelta().y() / 4
        size  = self.size
        self.sliders["size"].setValue(int(size - delta))

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key.Key_Space:
            self.change_status(not self.is_play)
        elif evt.key() == Qt.Key.Key_Right:
            self.next_frames()
        elif evt.key() == Qt.Key.Key_Left:
            self.prev_frames()

    def x_to_frame(self, x):
        x_play = self.playback.mapTo(self, QPoint(0, 0)).x()
        frame = int((x - x_play) // self.playback.bin)
        if frame > (self.n_frame - 1):
            frame = self.n_frame - 1
        return frame

    # DnD
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            for url in event.mimeData().urls():
                text = str(url.toLocalFile())
            self.load(path=text)
        else:
            event.ignore()


def reshape_matrix(mat_old, shp_new, dim_old, dim_new):
    """
    reshape (expand or shrink) the old matrix to new shape
    new matrix (mat_new) with data retained from the old matrix (mat_old)
    """
    mat_new = np.zeros(shp_new)
    if dim_new >= dim_old:
        mat_new[:, :dim_old] = mat_old[:, :dim_old]
    elif dim_new < dim_old:
        mat_new[:, :dim_new] = mat_old[:, :dim_new]
    return mat_new

def localtime():
    return time.strftime("%y%m%d%H%M%S", time.localtime())
