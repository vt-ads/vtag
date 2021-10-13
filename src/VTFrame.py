from lib import *

class VTFrame(QLabel):
    '''
    Will keep imgRaw, imgVis and imgQmap
    '''

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.pixmap = None
        self.img_detect = None
        self.show_detect = True
        self.cx = -20
        self.cy = -20
        # mouse moving events
        self.mx = -1
        self.my = -1
        # ground truth
        self.alpha = 200
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


def getIdx8QImg(img, k, alpha=200):  # k=20
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
        # use '%' to iterate the colormap
        qImg.setColor(i + 1, colormap[(i % nc) + 1].rgba())
    return QPixmap(qImg)


def getGrayQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format_Grayscale8)
    return QPixmap(qImg)
