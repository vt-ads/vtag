# import imageio
import os
import sys
import time
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster    import AgglomerativeClustering
from sklearn.neighbors  import kneighbors_graph
from sklearn.mixture    import GaussianMixture
from PyQt5.QtGui        import QPixmap, QImage, QPaintDevice, QPainter, qRgb
from PyQt5.QtWidgets    import (QApplication, QPushButton, QWidget, QLabel, QSlider,
                                QGridLayout,  QVBoxLayout, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore       import Qt, QTimer, QObject, QThread, pyqtSignal


def get_binary(imgs, f0, f1, cut=0.5, upbound=255):
    """
    Turn the iamge to 0 or upbound by the threashold cut
    """
    if (f0 < 0) or (f1 >= len(imgs)):
        out = np.zeros(imgs.shape[1:3])
    else:
        out_std = imgs[f0:f1].std(axis=(0, ))
        # out_std[out_std < np.quantile(out_std, .9)] = 0
        out_std = (out_std - out_std.min()) * upbound / \
            (out_std.max() - out_std.min())
        thresh = upbound * cut
        _, out = cv.threshold(out_std, thresh, upbound, cv.THRESH_BINARY)

    return out.astype(np.uint8)


def turn_img_to_clusters(img, k):
    """
    turn bitmap to x, y vector/array
    """
    y, x = np.nonzero(img)
    img2d = np.array([[y[i], x[i]] for i in range(len(x))])

    # clustering
    knn_graph = kneighbors_graph(img2d, 30, include_self=False)
    model = AgglomerativeClustering(linkage="ward",  # average, complete, ward, single
                                    connectivity=knn_graph,
                                    n_clusters=k)
    model.fit(img2d)

    # get output labels
    lbs = model.labels_
    # plt.scatter(img2d[lbs == 0, 1], img2d[lbs == 0, 0], )
    # plt.scatter(img2d[lbs == 1, 1], img2d[lbs == 1, 0], )

    # create labeled image
    img_c = np.zeros(img.shape)
    for i, j, k in zip(y, x, lbs):
        img_c[i, j] = k + 1

    return img_c


def detect_imgs(imgs, frame, span=10):
    ls_imgs = [get_binary(imgs, frame - span, frame + 1, upbound=1),
            #    get_binary(imgs, frame - span - 1, frame + 1, upbound=1),
            #    get_binary(imgs, frame, frame + span + 2, upbound=1),
               get_binary(imgs, frame, frame + span + 1, upbound=1)]
    out = (np.sum(ls_imgs, axis=0) / len(ls_imgs)).astype(np.uint8)
    _, out = cv.threshold(out, .5, 1, cv.THRESH_BINARY)
    # out[out < 1.5] = 0
    return out


def np2qt(img):
    height, width = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImg


def ls_files(path):
    ls_imgs = os.listdir(path)
    ls_imgs.sort()
    return ls_imgs


def load_np(files, n_imgs=-1):
    if n_imgs == -1:
        n_imgs = len(files)
    h, w, c = cv.imread(files[0]).shape

    imgs_rgb = np.zeros((n_imgs, h, w, c), dtype=np.uint8)
    for i in range(n_imgs):
        imgs_rgb[i] = cv.imread(files[i])
    imgs_bw = imgs_rgb.sum(axis=3)

    return imgs_bw
