# import imageio
import os
import sys
import time
import copy
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
from scipy.signal      import spline_filter
from scipy.signal      import convolve
from scipy.signal      import convolve2d
from scipy.signal      import find_peaks
from skimage.measure   import block_reduce
from sklearn.cluster   import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture   import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PyQt5.QtGui import QPixmap, QImage, QPaintDevice, QPainter, qRgb, QColor, QPen
from PyQt5.QtWidgets   import (QApplication, QPushButton, QWidget, QLabel, QSlider,
                                QGridLayout,  QVBoxLayout, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore      import Qt, QTimer, QObject, QThread, pyqtSignal

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


def detect_imgs(imgs, frame, span=1):
    i = frame
    j = span
    # stack 2*2 iamges: out_img is a 4-frame std image
    out_std = []
    for _ in range(2):
        out_std += [
            imgs[i:  (i+j+1)].std(axis=(0, )),
            imgs[(i-j):(i+1)].std(axis=(0, ))
        ]
        j += 1

    out_img = sum(out_std) / len(out_std)
    # cutoff  = np.median(out_img) + (n_sd * np.std(out_img))
    # out     = get_binary(out_img, cutabs=cutoff)
    return out_img

# === === === === === === === QT === === === === === === ===

def np2qt(img):
    height, width = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImg


def ls_files(path):
    ls_imgs = os.listdir(path)
    ls_imgs.sort()
    return [f for f in ls_imgs if ".png" in f]


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


def do_k_means(imgs, i, k):
    """
    imgs: series number of images (video)
    i: i frame image
    k: number of k of clustering
    """
    # ## Version one: only yx coordinate
    # pos_yx = find_nonzeros(imgs[i])

    ## Version two: involve temporal, neighbor pixels, and yx coordinate
    pos_yx = find_nonzeros(imgs[i])
    # n: number of nonzero pixels
    n = len(pos_yx)
    # computer feature length: tp(8) + sp(4) + pos(2)
    feature_length = 8 + 4 + 2
    # pre-allocate data space
    dt_features = np.zeros((n, feature_length), dtype=np.int)

    for j in range(n):
        # (y, x) coordinate
        pos_y, pos_x = pos_yx[j]
        # compute data cube for spatial or temporal analysis
        block_sp = make_block(imgs, i, (pos_y, pos_x), size=(3, 3))
        block_tp = make_block(imgs, i, (pos_y, pos_x), size=(2, 2, 2))
        # if out of boundary, skip the rest steps
        if (len(block_sp) == 0) or (len(block_tp) == 0):
            continue
        else:
            # extract features from blocks
            ft_tp = extract_features(block_tp, conv_type="temporal")
            ft_sp = extract_features(block_sp, conv_type="spatial")
            # concatenate features
            dt_features[j] = np.concatenate([ft_tp, ft_sp, pos_yx[j]])
    # remove out-of-boundary entry
    dt_features = dt_features[np.sum(dt_features, axis=1) != 0]
    # run k-mean by openCV
    if (len(dt_features) >= k):
        clusters, centers = cv_k_means(dt_features, k)
        return clusters, centers, dt_features[:, -2:] # last 2 are yx coordinates
    else:
        return -1


def cv_k_means(data, k):
    data = data.astype(np.float32)
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

def distance(pt1, pt2, is_dir=False):
    direction = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    distance  = (direction[0] ** 2 + direction[1] ** 2) ** .5

    if is_dir:
        return distance, direction
    else:
        return distance

def sort_by_dist(OUT, i):
    '''
    Sort k centers in cts[i + 1] based on distance between (i) and (i + 1)
    args
        i  : ith frame
    Time complexity O = (k-1)^2
    '''
    k = OUT["k"]
    if i == (len(OUT["centers"]) - 1):
        return OUT

    ls_min = []
    # 0-0, 0-1, 0-2, 1-0, 1-1
    for k1 in range(0, k - 1):
        ls_dist = []
        for k2 in range(k1, k):
            ls_dist += [distance(OUT["centers"][i][k1],
                                 OUT["centers"][i + 1][k2])]
        # find min idx
        val_min = np.min(ls_dist)
        ls_min  += [val_min]
        idx_min = np.where(ls_dist == val_min)[0][0] + k1
        # swap centers at i+1
        tmp                            = OUT["centers"][i + 1][k1].copy()
        OUT["centers"][i + 1][k1]      = OUT["centers"][i + 1][idx_min]
        OUT["centers"][i + 1][idx_min] = tmp
        # swap clustering i+1
        if OUT["clusters"][i + 1] is not None:
            pos_k1  = OUT["clusters"][i + 1] == k1
            pos_min = OUT["clusters"][i + 1] == idx_min
            OUT["clusters"][i + 1][pos_k1]   = idx_min
            OUT["clusters"][i + 1][pos_min]  = k1

    # compute the distance of the last k
    ls_min += [distance(OUT["centers"][i][k - 1],
                        OUT["centers"][i + 1][k - 1])]
    # return
    OUT["distance"][i]   = ls_min
    OUT["directions"][i] = OUT["centers"][i + 1] - OUT["centers"][i]
    return OUT

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


def make_block(inputs, i, pos, size=(3, 3)):
    """
    inputs: n x h x w, n is number of frames
    i: which focus frame
    pos: (x, y)
    """
    pos_y, pos_x = pos
    if len(size) == 3:
        bin_t, bin_y, bin_x = size
        block = inputs[(i - bin_t): (i + bin_t + 1),
                       (pos_y - bin_y): (pos_y + bin_y + 1),
                       (pos_x - bin_x): (pos_x + bin_x + 1)]
    else:
        bin_y, bin_x = size
        block = inputs[i,
                       (pos_y - bin_y): (pos_y + bin_y + 1),
                       (pos_x - bin_x): (pos_x + bin_x + 1)]
    return block


def extract_features(block, conv_type="temporal"):
    """
    Return: 8 element features (temporal), or 4 element features (spatial)
    """
    if conv_type == "temporal":
        # define kernel
        kernel_t = np.array(
            ([[[-1, -2, -1],
               [-2, -4, -2],
               [-1, -2, -1]],
              [[2, 4, 2],
               [4, 8, 4],
               [2, 4, 2]],
              [[-1, -2, -1],
               [-2, -4, -2],
               [-1, -2, -1]]]),
            dtype='int')
        # convolution
        block_con = convolve(block, kernel_t, mode='same')
        # max pooling
        block_pool = block_reduce(block_con, (3, 3, 3), np.max)

    elif conv_type == "spatial":
        # define kernel
        kernel_s = np.array(
            ([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]]),
            dtype='int')
        # convolution
        block_con = convolve(block, kernel_s, mode='same')
        # max pooling
        block_pool = block_reduce(block_con, (5, 5), np.max)

    # flatten
    block_1d = block_pool.reshape((-1))
    # return
    return block_1d
