# basic
import os
import sys
import time
import copy
import pickle
# cv2
import cv2 as cv
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
# pyqtgraph
import pyqtgraph as pg
# pandas
import pandas as pd
# numpy
import numpy  as np
# scipy
from scipy.signal import spline_filter, convolve, convolve2d, find_peaks
from scipy.stats import pearsonr
# skimage
from skimage.measure       import block_reduce
# sklearn
from sklearn               import cluster, datasets, mixture
from sklearn.cluster       import AgglomerativeClustering
from sklearn.neighbors     import kneighbors_graph
from sklearn.mixture       import GaussianMixture
from sklearn.linear_model  import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# pyqt5
from PyQt5.QtGui     import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore    import *

# === === === === === === === QT === === === === === === ===

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
    return out_img


def load_labels(n_frames, k):
    try:
        labels = pd.read_csv("labels.csv")
        labels = lb_from_pd_to_np(labels)
    except:
        labels = np.zeros((n_frames, k, 2), dtype=np.int)

    return labels


def make_labels(imgs_p):
    n_frames = len(imgs_p)
    n_ids = np.max(imgs_p)

    labels = np.zeros((n_frames, n_ids, 2), dtype=np.int)

    for i in range(n_frames):
        for ki in range(n_ids):
            y, x = np.nonzero(imgs_p[i] == (ki + 1))
            if (len(y) > 0) and (len(x) > 0):
                labels[i, ki] = np.median(y), np.median(x)
    return labels


def lb_from_np_to_pd(labels):
    n = len(labels)
    return np.array(labels).reshape((n, -1))


def lb_from_pd_to_np(labels):
    n = len(labels)
    return np.array(labels).reshape((n, -1, 2))


def sort_clusters(clusters_inputs, imgs):
    clusters = clusters_inputs.copy()
    n_frames, k, _ = clusters.shape
    for i in range(n_frames - 1):  # the last frame is not applicable
        img = imgs[i + 1]
        # 0-0, 0-1, 0-2, 1-0, 1-1
        for k1 in range(0, k - 1):
            ls_dist = []

            for k2 in range(k1, k):
                ls_dist += [distance(clusters[i][k1],
                                     clusters[i + 1][k2])]
            # find min idx
            idx_min = np.where(ls_dist == np.min(ls_dist))[0][0] + k1

            # swap centers at i+1
            tmp = clusters[i + 1][k1].copy()
            clusters[i + 1][k1] = clusters[i + 1][idx_min]
            clusters[i + 1][idx_min] = tmp

            # swap image values
            value_min = idx_min + 1  # 0 -> 1, 1 -> 2
            value_k1  = k1 + 1
            img[img == value_min] = 9
            img[img == value_k1]  = value_min
            img[img == 9]        = value_k1

    return clusters


# === === === === === === === QT === === === === === === ===

def np2qt(img):
    height, width = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImg


def ls_files(path="."):
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


def do_k_means(imgs, pos_yx, i, k):
    """
    imgs: series number of images (video)
    edges: points (y, x) of detected pixels
    i: i frame image
    k: number of k of clustering
    """
    # n: number of nonzero pixels
    n = len(pos_yx)
    # computer feature length: tp(8) + sp(4) + pos(12)
    feature_length = 24
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
            n_conv = len(ft_tp) + len(ft_sp)
            ft_yx = list(pos_yx[j]) * (n_conv // 2) # make yx same length as conv
            dt_features[j] = np.concatenate([ft_tp, ft_sp, ft_yx])
    # remove out-of-boundary entry
    idx_keep = np.sum(dt_features, axis=1) != 0
    dt_features = dt_features[idx_keep]
    yx = pos_yx[idx_keep]
    # scale features
    dt_features = scale2D(dt_features)
    # run k-mean by openCV
    if (len(dt_features) >= k):
        clusters, _ = cv_k_means(yx, k)
        centers = np.zeros((k, feature_length))
        for i in range(k):
            centers[i] = np.mean(dt_features[clusters == i], axis=0)

        return clusters, centers, yx  # last 2 are yx coordinates
    else:
        return -1


def cluster_by_coordinates(imgs, i, k):
    """
    imgs: series number of images (video)
    i: i frame image
    k: number of k of clustering
    """
    pos_yx = find_nonzeros(imgs[i])
    n = len(pos_yx)



def cluster_by_structure(data, yx, k):
    # collect info
    n, p = data.shape
    n_neighbor = int(len(data) * .1)
    # define structure
    structure = kneighbors_graph(yx, n_neighbor)
    # clustering
    model     = AgglomerativeClustering(linkage="ward",  # average, complete, ward, single
                                        connectivity=structure,
                                        n_clusters=k)
    model.fit(data)
    # get output labels
    clusters = model.labels_
    # compute centers
    centers  = np.zeros((k, p))
    for i in range(k):
        centers[i] = np.mean(data[clusters == i], axis=0)
    # return
    return clusters, centers


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


def filter_edges(edgs, bounds):
    paths = mpath.Path(bounds)
    idx_keep = paths.contains_points(edgs)
    return edgs[idx_keep]



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
              [[ 2,  4,  2],
               [ 4,  8,  4],
               [ 2,  4,  2]],
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


def scale2D(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def map_features_to_id(features, k, use_pca=False):
    n_ft = len(features)
    new_ids = np.array([0] * n_ft)

    if np.mean(features) == 0:
        fake_pcs = np.zeros((len(features), 2))
        return new_ids, fake_pcs

    else:
        #-- Get PCs from features, and cluster into k+1 groups
        pca = PCA(2)
        pca.fit(features) 
        # pcs = pca.transform(features) * pca.explained_variance_ratio_
        pcs = pca.transform(features)
        # ids, _ = cv_k_means(pcs, k)
        ids = cluster_gm(pcs, k)

        # return
        return ids + 1, pcs


def reassign_id_by(old_ids, values=None, by="size"):
    # pre-allocate new ids
    new_ids = np.array([0] * len(old_ids))

    # get collection of cluster properties
    n_ids = len(old_ids)
    value_counts = pd.value_counts(old_ids)
    keys = value_counts.keys()

    # two strategies
    if by == "size":
        "Cluster with smallest size will be assigned to '0'"
        # find which key occur minimum
        idx_remove = np.where(value_counts == np.min(value_counts))[0][0]

    elif by == "distance":
        "Cluster with largest averaged distance from its center will be assigned to '0'"
        "Also consider size"
        dist = []
        for i in keys:
            # computer cluster center
            pts = values[old_ids == i]
            pt_ct = np.median(pts, axis=0)
            # calculate averaged distance
            dist_pts = []
            for pt in pts:
                dist_pts += [distance(pt, pt_ct)]
            dist += [np.mean(dist_pts)]

        # find which key has largest distance and smallest numbers
        scores = -np.array(dist) * (n_ids - value_counts)
        print("keys: ", keys)
        print("dist: ", dist)
        print("counts: ", value_counts)
        print("scores: ", scores)
        idx_remove = np.where(scores == np.min(scores))[0][0]

    elif by == "value":
        "Cluster with largest values will be assigned to '0'"
        med_values = [np.median(values[old_ids == i]) for i in keys]
        idx_remove = np.where(med_values == max(med_values))[0][0]

    # find which clusters to keep
    key_remove = keys[idx_remove]
    key_rest   = keys[keys != key_remove]

    # re-assign: 0-> background, others-> majority
    for i in range(len(key_rest)):
        assign_key = key_rest[i]
        assign_pos = np.where(old_ids == assign_key)[0]
        new_ids[assign_pos] = i + 1


    return new_ids


def remove_outliers(pts, out_std=2):
    n_pts = len(pts)

    if n_pts > 2:
        idx_all   = np.array(range(n_pts))
        bool_all  = np.array([False] * n_pts)
        idx_keep  = idx_all
        bool_keep = bool_all
        while not all(bool_keep):
            dist = []
            for i in idx_keep:
                idx = [k for k in range(n_pts) if k != i]
                a, b, c = fit_linear(pts[idx])
                dist += [distance_to_line(a, b, c, pts[i, 0], pts[i, 1])]
            bool_keep = dist < np.median(dist) + np.std(dist) * out_std
            idx_keep = idx_keep[bool_keep]

        bool_all[idx_keep] = True
        return pts[idx_keep], bool_all

    else:
        return pts, np.array([True] * n_pts)


def cluster_gm(data, k, weights=None):
    gm = GaussianMixture(n_components=k,
                         max_iter=100,
                         n_init=10,
                         weights_init=weights,
                        init_params="kmeans",
                        tol=1e-4)
    return gm.fit_predict(data)

def fit_linear(pts):
    """
    pts[:, 0]: X
    pts[:, 1]: y
    ax + by + c = 0
    """
    reg = LinearRegression().fit(pts[:, 0].reshape(-1, 1), pts[:, 1])
    slope = reg.coef_[0]
    intercept = reg.intercept_
    a = slope
    b = -1
    c = intercept

    return a, b, c,


def distance_to_line(a, b, c, x, y):
    return np.abs(a*x + b*y + c) / (a**2 + b**2)**.5


def save_labels(labels, n_ids, file="labels.csv"):
    colnames = np.array([["k%d_y" % i, "k%d_x" % i]
                         for i in range(n_ids)]).reshape(-1)
    labels = lb_from_np_to_pd(labels)
    dt = pd.DataFrame(labels)
    dt.columns = colnames
    dt.to_csv(file, index=False)
