from PyQt5.QtGui import QCursor
from lib import *
from Tags import VTags
from Playback import Playback

colorsets = np.array(["#000000",
                      "#ffff33", "#f94144", "#f3722c", "#f8961e", "#f9844a",
                      "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590",
                      "#277da1"])

class Player(QWidget):
    def __init__(self, folder="/Users/jchen/Dropbox/projects/Virtual_Tags/data"):
        super().__init__()
        self.setMouseTracking(True)

        # WD
        dataname = "group_small"
        dataname = "group"
        os.chdir(folder)
        os.chdir(dataname)

        # Predictions
        self.imgs_show = []
        # Frames
        self.folder  = folder
        self.paths   = ls_files()
        self.playback = Playback()
        self.frame   = QFrame()
        self.plot    = pg.plot()
        self.n_frame = 0
        self.i_frame = 0
        self.lb_frame = QLabel("Frame: %d" % self.i_frame)
        self.fps      = 12.5 / 1000
        # Predictions
        self.sli_thre = QSlider(Qt.Horizontal, self)
        self.lb_thre  = QLabel("Threshold: %.3f" % (self.sli_thre.value()/1000))
        self.sli_span = QSlider(Qt.Horizontal, self)
        self.lb_span  = QLabel("Span: %d" % self.sli_span.value())
        # Status
        self.is_play = False
        self.label_counter = 0
        # Setup timer
        self.timer  = QTimer(self)

        # GUI
        self.buttons = dict(browse    = QPushButton("Browse"),
                            show_lbs  = QPushButton("Hide Labels"),
                            show_pred = QPushButton("Hide Predictions"),
                            play  = QPushButton("Play"),
                            next  = QPushButton("Next frame > "),
                            prev  = QPushButton("< Previous frame"),
                            save  = QPushButton("Save labels"))
        self.toggles = dict(edges = QRadioButton("Edges"),
                            cls   = QRadioButton("Clusters"),
                            pre   = QRadioButton("Predictions"))
        self.globalrec = dict(frame = QRect(0, 0, 0, 0),
                              play  = QRect(0, 0, 0, 0))

        # init
        self.load_VTags()
        self.init_runtime()
        self.initUI()
        self.update_frames()


    def load_VTags(self):
        self.ARGS, self.IMGS, self.OUTS = pickle.load(open("model.h5", "rb"))
        self.toggles["pre"].setChecked(True)
        self.imgs_show = self.IMGS["pred"]  # define what show on the screen
        self.n_frame   = self.ARGS["n"]
        self.playback.set_n(self.n_frame)
        if self.ARGS["n_id"] == 2:
            pre_grp = np.array(pd.read_csv("labels.csv")).reshape((self.n_frame, 2, 2))
            dist    = np.array([distance(p1, p2) for p1, p2 in pre_grp])
            self.playback.set_heat(dist)
        # try load existing labels
        try:
            labels = pd.read_csv("labels.csv")
            self.OUTS["pred_labels"] = lb_from_pd_to_np(labels)
        except Exception as e:
            print(e)

    def init_runtime(self):
        self.timer.timeout.connect(self.next_frames)
        self.buttons["show_lbs"].clicked.connect(self.toggle_lbs)
        self.buttons["show_pred"].clicked.connect(self.toggle_pred)
        self.buttons["play"].clicked.connect(
            lambda x: self.change_status(not self.is_play))
        self.buttons["next"].clicked.connect(self.next_frames)
        self.buttons["prev"].clicked.connect(self.prev_frames)
        self.buttons["save"].clicked.connect(self.save_lbs)
        self.toggles["edges"].clicked.connect(self.toggle)
        self.toggles["cls"].clicked.connect(self.toggle)
        self.toggles["pre"].clicked.connect(self.toggle)
        self.sli_span.valueChanged.connect(self.change_span)
        self.sli_thre.valueChanged.connect(self.change_thre)

    def toggle_lbs(self):
        status = not self.frame.show_lbs
        if status:
            self.buttons["show_lbs"].setText("Hide Labels")
        else:
            self.buttons["show_lbs"].setText("Show Labels")
        self.frame.show_lbs = status
        self.update_frames()

    def toggle_pred(self):
        status = not self.frame.show_detect
        if status:
            self.buttons["show_pred"].setText("Hide Predictions")
        else:
            self.buttons["show_pred"].setText("Show Predictions")
        self.frame.show_detect = status
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
        n_ids = self.ARGS["n_id"]
        save_labels(labels, n_ids, "labels.csv")

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

        # Layout
        layout = QGridLayout(self)
        self.buttons["save"].setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)
        self.frame.setSizePolicy(QSizePolicy.Expanding,
                                 QSizePolicy.Expanding)
        self.playback.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
        self.frame.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # layout.addWidget(self.buttons["browse"], 0, 0, 1, 4)
        layout.addWidget(self.frame,             1, 0, 1, 3, alignment=Qt.AlignLeft)
        layout.addWidget(self.plot,              1, 3, 1, 2, alignment=Qt.AlignRight)
        layout.addWidget(self.lb_frame,          2, 0, 1, 1)
        # layout.addWidget(self.playback,          3, 0, 1, 3)
        layout.addWidget(self.playback,        3, 0, 1, 3)
        layout.addWidget(self.toggles["edges"],  4, 0, 1, 1)
        layout.addWidget(self.toggles["cls"],    4, 1, 1, 1)
        layout.addWidget(self.toggles["pre"],    4, 2, 1, 1)
        layout.addWidget(self.buttons["prev"],   5, 0)
        layout.addWidget(self.buttons["play"],   5, 1)
        layout.addWidget(self.buttons["next"],   5, 2)
        layout.addWidget(self.buttons["show_lbs"],   2, 3, 2, 1)
        layout.addWidget(self.buttons["show_pred"],  2, 4, 2, 1)
        layout.addWidget(self.buttons["save"],       4, 3, 2, 2)

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
        self.lb_frame.setText("Frame: %d" % i)
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
        self.show_lbs = True
        self.lb_x = []
        self.lb_y = []
        # label counter
        self.label_counter = 0

    def set_image(self, pixmap):
        self.pixmap = pixmap

    def set_predict(self, img):
        self.img_detect = getIdx8QImg(img,  int(np.max(img)))

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


def getIdx8QImg(img, k): # k=20
    colormap = [QColor(c) for c in colorsets]

    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Indexed8)
    nc = len(colormap) - 1  # exclude 0: background color

    # background color
    # qImg.setColor(0, colormap[0].rgba())
    qImg.setColor(0, QColor(0, 0, 0, 255).rgba())
    qImg.setColor(0, QColor(0, 0, 0, 200).rgba())
    # cluster color
    for i in range(k):
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
