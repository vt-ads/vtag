import pandas    as pd
import numpy     as np
import cv2       as cv

def get_binary(signals, cut=.5, cutabs=None, upbound=1):
    """
    parameters
    ---
    cut   : Use the quantile of input data as the threshold
    cutabs: Use absolute values as the threshold

    return
    ---
    signals with values of [0 or upbound] in the same dimensions
    """
    if cutabs is None:
        # set color to 1 if over 50 percentile
        _, out = cv.threshold(signals,
                        np.quantile(signals, cut), upbound, cv.THRESH_BINARY)
    else:
        # set color to 1 if over cutabs
        _, out = cv.threshold(signals,
                        cutabs, upbound, cv.THRESH_BINARY)
    return out

def load_labels(n_frames, k):
    try:
        labels = pd.read_csv("labels.csv")
        labels = lb_from_pd_to_np(labels)
    # see "lb_from_pd_to_np"
    except:
        labels = np.zeros((n_frames, k, 2), dtype=np.int)
    # if there is no "labels.csv" in current directory
    # create "labels" as n*k*2
    # n: number of frames/pictures
    # k: number of animals
    return labels

def lb_from_np_to_pd(labels):
    ""
    n = len(labels)
    return np.array(labels).reshape((n, -1))
    # change the input into a long vector with length n


def lb_from_pd_to_np(labels):
    n = len(labels)
    return np.array(labels).reshape((n, -1, 2))

import os
import sys
import time
import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import pyqtgraph as pg
# scipy
from scipy.signal import spline_filter, convolve, convolve2d, find_peaks
from scipy.stats  import pearsonr
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

# === === === === === === === color === === === === === === ===

colorsets = np.array(["#000000",
                      "#ffff33", "#f94144", "#f3722c", "#f8961e", "#f9844a",
                      "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590",
                      "#277da1"])

alpha = 200
palette_viridis = [
    QColor(68, 1, 84, alpha),
    QColor(72, 40, 62, alpha),
    QColor(62, 74, 137, alpha),
    QColor(49, 104, 142, alpha),
    QColor(38, 130, 142, alpha),
    QColor(31, 158, 137, alpha),
    QColor(53, 183, 121, alpha),
    QColor(109, 205, 89, alpha),
    QColor(180, 222, 44, alpha),
    QColor(253, 231, 37, alpha)
]
palette_viridis.reverse()

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

# === === === === === === === QT === === === === === === ===




def make_labels(imgs_p):
    # imgs_p: n*h*w, animal label for interesting pixels
    n_frames = len(imgs_p)
    # number of frames
    n_ids = np.max(imgs_p)
    # number of animals

    labels = np.zeros((n_frames, n_ids, 2), dtype=np.int)
    # n*number of animals*2

    for i in range(n_frames):
        # for every frame
        for ki in range(n_ids):
            # for every animal
            y, x = np.nonzero(imgs_p[i] == (ki + 1))
            # the coordinates of pixels that are assigned to animals
            if (len(y) > 0) and (len(x) > 0):
                labels[i, ki] = np.median(y), np.median(x)
                # the median coordinates of pixels
    return labels


# ARCHIVE
# def sort_clusters(clusters_inputs, imgs):
#     clusters = clusters_inputs.copy()
#     n_frames, k, _ = clusters.shape
#     for i in range(n_frames - 1):  # the last frame is not applicable
#         img = imgs[i + 1]
#         # 0-0, 0-1, 0-2, 1-0, 1-1
#         for k1 in range(0, k - 1):
#             ls_dist = []

#             for k2 in range(k1, k):
#                 ls_dist += [distance(clusters[i][k1],
#                                      clusters[i + 1][k2])]
#             # find min idx
#             idx_min = np.where(ls_dist == np.min(ls_dist))[0][0] + k1

#             # swap centers at i+1
#             tmp = clusters[i + 1][k1].copy()
#             clusters[i + 1][k1] = clusters[i + 1][idx_min]
#             clusters[i + 1][idx_min] = tmp

#             # swap image values
#             value_min = idx_min + 1  # 0 -> 1, 1 -> 2
#             value_k1  = k1 + 1
#             img[img == value_min] = 9
#             img[img == value_k1]  = value_min
#             img[img == 9]        = value_k1
#     return clusters

def sort_clusters(clusters, imgs):
    # clusters: n*number of animals*2, location of predicated animals
    # imgs: n*h*w
    # the animal id for all of the interesting pixels in the image
    n_frames, k, _ = clusters.shape

    for i in range(n_frames):
        # for every frame
        img          = imgs[i]
        score_ori    = get_scores(clusters, i)
        clusters_alt = clusters.copy()

        for k1 in range(0, k - 1):
            for k2 in range(k1, k):
            # for every animal pair-wise comparison
                clusters_alt[i] = swap_clusters(clusters[i], swp1=k1, swp2=k2)
                score_alt = get_scores(clusters_alt, i)

                if score_alt > score_ori:
                    clusters  = clusters_alt.copy()
                    score_ori = score_alt
                    # update images
                    img[img == (k1 + 1)] = 9
                    img[img == (k2 + 1)] = k1 + 1
                    img[img == 9]        = k2 + 1
    return clusters
    # switch the animal label if they are flipped 


def get_scores(clts, n, weight=.7):
    # clts: n*number of animals*2, location of predicated animals 
    # n: index of frame
    try:
        vec_ori = clts[(n - 1): (n + 1)] - clts[(n - 2): n]
        # 2*number of animals*2

        # change of position
        vec = vec_ori[-1]
        # number of animals*2

        # change of direction
        dvec = vec_ori[1] - vec_ori[0] # ???
        # number of animals*2

        # output
        return -(euclidean(vec) + weight * euclidean(dvec))
    except:
        return 0


def euclidean(pts):
    return sum(np.sum(pts ** 2, axis=1) ** .5)


def swap_clusters(clusters, swp1=0, swp2=1):
    clusters_out = clusters.copy()
    tmp                = clusters_out[swp1].copy()
    clusters_out[swp1] = clusters_out[swp2].copy()
    clusters_out[swp2] = tmp
    return clusters_out

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
        kernel = np.array(([1] * 16), dtype='int') / 16

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
    # imgs: detected edges, n*h*w
    # pos_yx: number of non-zero pixels * 3
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
    # n*24

    for j in range(n):
    # for every non-zero pixel 
        # (y, x) coordinate
        pos_y, pos_x = pos_yx[j]
        # compute data cube for spatial or temporal analysis
        block_sp = make_block(imgs, i, (pos_y, pos_x), size=(3, 3))
        block_tp = make_block(imgs, i, (pos_y, pos_x), size=(2, 2, 2))
        # imgs: detected edges, n*h*w
        # for every non-zero pixel, 
        # get 7*7 spacial and 
        # 5*5*5 temporal block from it

        # if out of boundary, skip the rest steps
        if (len(block_sp) == 0) or (len(block_tp) == 0):
            continue
        # if true, go to the next j,
        # if false, go to else 
        else:
            # --- VERSION 1
            # extract features from blocks
            ft_tp = extract_features(block_tp, conv_type="temporal")
            ft_sp = extract_features(block_sp, conv_type="spatial")
            # for every corner of the spacial/temperal block, 
            # extract the local maxima 
            # 2*2 (4 maxima) from spacial block
            # 2*2*2 (8 maxima) from temporal block

            # concatenate features
            n_conv = len(ft_tp) + len(ft_sp)
            ft_yx = list(pos_yx[j]) * (n_conv // 2) 
            # make yx same length as conv
            # list(two elements) * 6, 12
            # repeat the coordinates of pixels 6 times 

            dt_features[j] = np.concatenate([ft_tp, ft_sp, ft_yx])
            # dt_features has 24 elements 
            # for every pixel of interest within every frame,
            # there are 24 elements of interest 
            # first 12 of which are edges (gray-scale color),
            # last 12 of which are 6 repeats of coordinates 

            # --- VERSION 2
            # dt_features[j] = list(pos_yx[j]) * 12

    # remove out-of-boundary entry
    idx_keep = np.sum(dt_features, axis=1) != 0
    # summation of all 24 elements should be non-zero
    # true and false with length number of non-zero pixels 
    dt_features = dt_features[idx_keep]
    yx = pos_yx[idx_keep]
    # scale features
    dt_features = scale2D(dt_features)
    # standardize all pixel's 24 features

    # run k-mean by openCV
    if (len(dt_features) >= k):
    # length of interesting pixels has to be larger than number of clusters 
        clusters, _ = cv_k_means(yx, k)
        # yx: coordinates of interesting pixels
        # k: number of clusters
        # clusters: labels with length of interesting pixels
        # for each pixel, a label of cluster is given
        centers     = np.zeros((k, feature_length))
        # k * 24
        for i in range(k):
        # for every cluster
            centers[i] = np.mean(dt_features[clusters == i], axis=0)
            # averaging across the features for the detected edges and coordinates for each cluster
            # this is different for the actual center from cv_k_means 

        return clusters, centers, yx  # last 2 are yx coordinates
        # clusters: number of interesting pixels * 1 (label of cluster)
        # centers: k*24 
        # yx: number of interesting pixels * 2 (coordinates of interesting pixels)
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
    # data: coordinates of interesting pixels
    # k: number of clusters
    data = data.astype(np.float32)
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # criteria: (1, 10, 1)
    param_k = dict(data=data,
                   K=k,
                   bestLabels=None,
                   criteria=criteria,
                   attempts=10,
                   flags=cv.KMEANS_PP_CENTERS)

    _, clusters, centers = cv.kmeans(**param_k)
    # output is compactness, labels, centers
    # compactness : the sum of squared distance from each point to their corresponding centers.
    # 1 value
    # labels : the label array 
    # number of pixels * 1
    # centers : array of centers of clusters.
    # k * 2, 2 coordinates 
    clusters = clusters.reshape(-1)
    # length is the number of pixels
    # value is the label of cluster, 0, 1, ...

    return clusters, centers
    # return label of each interesting pixel and center of clusters

def distance(pt1, pt2, is_dir=False):
    direction = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    distance  = (direction[0] ** 2 + direction[1] ** 2) ** .5

    if is_dir:
        return distance, direction
    else:
        return distance


def filter_edges(edgs, bounds):
    # edges: number of non-zero pixels by 3
    paths = mpath.Path(bounds)
    # draw lines
    idx_keep = paths.contains_points(edgs)
    # for the non-zero pixels within bounds
    return edgs[idx_keep]
    # return edges that are within bounds



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
        # 5 * 5 * 5 block
    else:
        bin_y, bin_x = size
        block = inputs[i,
                       (pos_y - bin_y): (pos_y + bin_y + 1),
                       (pos_x - bin_x): (pos_x + bin_x + 1)]
        # 7 * 7 block
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
        # for every 3 * 3 * 3 corner in temporal block, 
        # calculate the max 
        # output is 2 * 2 * 2 

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
        # for every 5*5 corner in spacial block, 
        # calculate the max 
        # output is 2 * 2 

    # flatten
    block_1d = block_pool.reshape((-1))
    # return
    return block_1d
    # return 4 or 8 element vector


def scale2D(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def map_features_to_id(features, k, use_pca=False):
    # features: k*24
    n_ft = len(features)
    # k
    new_ids = np.array([0] * n_ft)
    # 1*k

    if np.mean(features) == 0:
        fake_pcs = np.zeros((len(features), 2))
        # k*2
        return new_ids, fake_pcs

    else:
        # PCA clustering -----=-----
        #-- Get PCs from features, and cluster into k+1 groups
        pca = PCA(2)
        pca.fit(features) 
        # features: k*24
        # pcs = pca.transform(features) * pca.explained_variance_ratio_
        pcs = pca.transform(features)
        # pcs: k*2 
        ids, _ = cv_k_means(pcs, k)
        # ids = cluster_gm(pcs, k)
        # All clustering -----=-----
        # ids = cluster_gm(features, k)
        # pcs = np.zeros((len(features), 2))

        # return
        return ids + 1, pcs
        # ids+1: labels for each k, 1,2,3,..., pig number, k elements
        # pcs: 2 dimensional pc values 


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
