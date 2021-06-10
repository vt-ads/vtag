# import imageio
import os
import sys
import time
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import convolve, convolve2d

from sklearn.cluster    import AgglomerativeClustering
from sklearn.neighbors  import kneighbors_graph
from sklearn.mixture    import GaussianMixture
from PyQt5.QtGui        import QPixmap, QImage, QPaintDevice, QPainter, qRgb, QPen
from PyQt5.QtWidgets    import (QApplication, QPushButton, QWidget, QLabel, QSlider,
                                QGridLayout,  QVBoxLayout, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore       import Qt, QTimer, QObject, QThread, pyqtSignal


# def get_binary(signals, q):
#     """
#     Turn the iamge to 0 or upbound by the threashold cut
#     """
#     if (f0 < 0) or (f1 >= len(imgs)):
#         out = np.zeros(imgs.shape[1:3])
#     else:
#         out_std = imgs[f0:f1].std(axis=(0, ))
#         out_std[out_std < np.quantile(out_std, .9)] = 0
#         out_std = (out_std - out_std.min()) * upbound / \
#             (out_std.max() - out_std.min())
#         thresh = upbound * cut
#         _, out = cv.threshold(out_std, thresh, upbound, cv.THRESH_BINARY)

#     return out.astype(np.uint8)

def get_binary(signals, cut=.5, cutabs=None, upbound=1):
    if cutabs is None:
        _, out = cv.threshold(signals, np.quantile(
            signals, cut), upbound, cv.THRESH_BINARY)
    else:
        _, out = cv.threshold(signals, cutabs, upbound, cv.THRESH_BINARY)
    return out



def detect_imgs(imgs, frame, span=1, n_sd=3):
    i = frame
    j = span
    out_std = []
    for _ in range(2):
        out_std += [
            imgs[i:(i+j+1)].std(axis=(0, )),
            imgs[(i-j):(i+1)].std(axis=(0, ))
        ]
        j += 1

    out_img = sum(out_std) / len(out_std)
    cutoff = np.median(out_img) + (n_sd * np.std(out_img))
    _, out = cv.threshold(out_img, cutoff, 1, cv.THRESH_BINARY)
    # _, out = cv.threshold(sum(out_std), np.quantile(out_std, q), 1, cv.THRESH_BINARY))
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


def load_np(files, n_imgs=-1, is_BW=True):
    if n_imgs == -1:
        n_imgs = len(files)
    h, w, c = cv.imread(files[0]).shape

    imgs_rgb = np.zeros((n_imgs, h, w, c), dtype=np.uint8)
    for i in range(n_imgs):
        imgs_rgb[i] = cv.imread(files[i])

    if is_BW:
        imgs_bw = imgs_rgb.sum(axis=3)
        return imgs_bw
    else:
        return imgs_rgb


def find_clusters(img, k):
    """
    turn bitmap to x, y vector/array
    """
    img2d = find_nonzeros(img)

    # clustering
    knn_graph = kneighbors_graph(img2d, 30, include_self=False)
    model     = AgglomerativeClustering(linkage="ward",  # average, complete, ward, single
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


def find_center(img):
    img2d = find_nonzeros(img)
    try:
        cy, cx = np.median(img2d, axis=0)
    except Exception:
        cy, cx = -1, -1
    return cy, cx


def find_nonzeros(img, is_values=False):
    y, x = np.nonzero(img)
    if is_values:
        return np.array([[y[i], x[i], img[y[i], x[i]]] for i in range(len(x))])
    else:
        return np.array([[y[i], x[i]] for i in range(len(x))])

def get_k_centers(img, k):
    img_c = find_clusters(img, k)
    img2d = find_nonzeros(img_c, is_values=True)
    dt = pd.DataFrame(img2d)
    dt.columns = ["y", "x", "k"]
    return dt.groupby(["k"]).aggregate("median").reset_index()


def drawCross(x, y, painter, size=2):
    l1_st_x, l1_st_y = x-size, y-size
    l1_ed_x, l1_ed_y = x+size, y+size
    l2_st_x, l2_st_y = x-size, y+size
    l2_ed_x, l2_ed_y = x+size, y-size
    painter.drawLine(l1_st_x, l1_st_y, l1_ed_x, l1_ed_y)
    painter.drawLine(l2_st_x, l2_st_y, l2_ed_x, l2_ed_y)


def smooth_signals(signals, n, kernel=-1):
    if kernel == -1:
        # kernel = np.array(([1, 2, 4, 8, 12, 16, 12, 8, 4, 2, 1]), dtype='int') / 70
        kernel = np.array(
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype='int') / 16

    for _ in range(n):
        signals = convolve(signals, kernel, mode='same')

    return signals


def process_signals(signals, n_smth=5, cut_sd=2, burnin=16):
    signals_sm = smooth_signals(signals, n=n_smth)
    signals_abs = np.abs(signals - signals_sm)
    signals_fx = np.array(signals.copy())

    std = np.std(signals_abs[burnin:])
    is_out = signals_abs > np.median(signals_abs[burnin:]) + std * cut_sd
    is_out[:burnin] = False
    idx_out = np.where(is_out)[0]

    signals_fx[idx_out] = signals_sm[idx_out]

    return signals_fx


def do_k_means(img, k):
    np_yx = find_nonzeros(img).astype(np.float32)
    clusters, centers = cv_k_means(np_yx, k)
    return clusters, centers, np_yx

def cv_k_means(data, k):
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    param_k = dict(data=data,
                   K=k,
                   bestLabels=None,
                   criteria=criteria,
                   attempts=10,
                   flags=cv.KMEANS_PP_CENTERS)

    _, clusters, centers = cv.kmeans(**param_k)
    clusters = clusters.reshape(-1)

    return clusters, centers

def distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5

def sort_by_dist(cts, cls, i, k):
    '''
    Sort k centers in cts[i + 1] based on distance between (i) and (i + 1)
    args
        cts: list of centers
        cls: list of clusters
        i  : ith frame
        k  : number of clusters
    Time complexity O = (k-1)^2
    '''
    if i == (len(cts) - 1):
        return cts, cls, k * [0]

    ls_min = []
    # 0-0, 0-1, 0-2, 1-0, 1-1,
    for k1 in range(0, k - 1):
        ls_dist = []
        for k2 in range(k1, k):
            ls_dist += [distance(cts[i][k1], cts[i + 1][k2])]

        # find min idx
        val_min = np.min(ls_dist)
        ls_min  += [val_min]
        idx_min = np.where(ls_dist == val_min)[0][0] + k1
        # swap centers position
        tmp                 = cts[i + 1][k1].copy()
        cts[i + 1][k1]      = cts[i + 1][idx_min]
        cts[i + 1][idx_min] = tmp
        # swap clustering results
        if cls[i + 1] is not None:
            pos_k1 = cls[i + 1] == k1
            pos_min = cls[i + 1] == idx_min
            cls[i + 1][pos_k1] = idx_min
            cls[i + 1][pos_min] = k1

    # compute the distance of the last k
    ls_min += [distance(cts[i][k - 1], cts[i + 1][k - 1])]
    # return
    return cts, cls, ls_min

# No swap
# def sort_by_dist(cts, i, k):
#     '''
#     Sort k centers in cts[i + 1] based on distance between (i) and (i + 1)
#     Time complexity O = (k-1)^2
#     '''
#     if i == (len(cts) - 1):
#         return cts, k * [0]

#     ls_min = []
#     for k1 in range(k):
#         ls_dist = []
#         for k2 in range(k):
#             ls_dist += [distance(cts[i][k1], cts[i + 1][k2])]
#     # return
#     return ls_min
