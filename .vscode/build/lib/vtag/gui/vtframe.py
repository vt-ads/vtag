import numpy as np
from PyQt6.QtCore    import Qt, QRect
from PyQt6.QtGui     import QPainter, QPen, QBrush, QPixmap, QColor, QImage, QCursor, QFont
from PyQt6.QtWidgets import QLabel, QWidget

# vtag imports
from .colors        import vtcolor, ls_colors
from .utils         import drawCross

class VTFrame(QLabel):
    '''
    display centroids and contour
    '''

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.image = None

        # mouse moving events
        self.mx = -1
        self.my = -1
        self.size_m = 70 # detection area

        # vtag objects
        self.lbs = None
        self.poi = None

        # user options
        self.alpha    = 200   # opacity of the annotation
        self.show_lbs = False # whether to show labels (centroids)
        self.show_poi = False # whether to show poi (motino)

        # idx which label is assgined (for cursor)
        self.i_tag = 0

    def set_labels(self, labels):
        """
        should be vtag.DATA["track"][i], a dataframe
        """
        self.lbs = labels

    def set_poi(self, poi):
        self.poi = poi

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_image(self, image):
        """
        parameters
        ---
        image: numpy 3D-array for the RGB image (w, h, c)
        """
        try:
            self.image = getGrayQImg(image)
        except Exception as e:
            print("vtframe: ", e)
            self.image = None

    def paintEvent(self, event):
        super().paintEvent(event)
        # open painter
        painter = QPainter(self)

        # ---draw background image
        if self.image is not None:
            self.setPixmap(self.image)
            # annotation opacity
            w, h = self.image.width(), self.image.height()
            mask = QPixmap(w, h)
            mask.fill(QColor(0, 0, 0, self.alpha))
            painter.drawPixmap(0, 0, mask)

        # ---draw poi
        if self.show_poi:
            draw_poi(self.poi, painter=painter, alpha=self.alpha)

        # ---draw labels
        if self.show_lbs:
            draw_labels(labels=self.lbs, size=self.size_m, painter=painter)

        # close painter
        painter.end()

        # cursor
        cursor = get_QCursor(i=self.i_tag, size=self.size_m)
        self.setCursor(cursor)

    def mouseMoveEvent(self, evt):
        pt = evt.position().toPoint()
        self.mx, self.my = pt.x(), pt.y()


def get_QCursor(i, size=15):
    # create transparent background
    canvas = QPixmap(size, size)
    canvas.fill(QColor(0, 0, 0, 0))

    # create painter
    painter = QPainter(canvas)
    painter.drawPixmap(0, 0, canvas)

    # draw circles
    brush = QBrush()
    brush.setStyle(Qt.BrushStyle.SolidPattern)
    draw_circle(painter, brush, 0, 0, size, i)

    # close painter
    painter.end()
    # return
    return QCursor(canvas)

def draw_poi(poi, painter, alpha=200):
    """
    poi should be a 2D binary mask with the same dimension
    """
    pixmap = getIdx8QImg(poi, k=9, alpha=alpha)
    painter.drawPixmap(0, 0, pixmap)

def draw_labels(labels, size, painter):
    brush = QBrush()
    brush.setStyle(Qt.BrushStyle.SolidPattern)
    for i, (x, y) in enumerate(labels):
        draw_circle(painter, brush, x - size / 2, y - size / 2, size, i)
        rect = QRect(x + 5, y + 5, 10, 10)
        font = QFont()
        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(rect, 0, "%d" % i)

def draw_circle(painter, brush, x, y, size, i_color, alpha=70):
    """
    x, y are at the top-left already
    """
    # centroids
    brush.setColor(vtcolor(i_color + 1))
    painter.setBrush(brush)
    size_ct = 10
    painter.drawEllipse(x + (size / 2) - (size_ct / 2),
                        y + (size / 2) - (size_ct / 2),
                        size_ct, size_ct)
    # detect area
    brush.setColor(vtcolor(i_color + 1, alpha=alpha))
    painter.setBrush(brush)
    painter.drawEllipse(x, y, size, size)


def getRGBQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w, h, w*3, QImage.Format.Format_RGB888)
    return QPixmap(qImg)

def getBinQImg(img, alpha=150):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format.Format_Indexed8)
    qImg.setColor(0, QColor(0, 0, 0, 0).rgba())
    qImg.setColor(1, QColor(255, 255, 51, alpha).rgba())
    return QPixmap(qImg)

def getIdx8QImg(img, k, alpha=200):  # k=20
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w * 1, QImage.Format.Format_Indexed8)
    nc = len(ls_colors) - 1  # exclude 0: background color
    # background color
    qImg.setColor(0, QColor(0, 0, 0, alpha).rgba())
    # cluster color
    for i in range(k):
        # use '%' to iterate the colormap
        qImg.setColor(i + 1, vtcolor((i % nc) + 1).rgba())
    return QPixmap(qImg)

def getGrayQImg(img):
    h, w = img.shape[0], img.shape[1]
    qImg = QImage(img.astype(np.uint8).copy(), w,
                  h, w*1, QImage.Format.Format_Grayscale8)
    return QPixmap(qImg)
