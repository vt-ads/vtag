# import pickle
import os
import numpy as np
import cv2   as cv
from scipy.signal import spline_filter, convolve, convolve2d, find_peaks


# from .lib import get_binary

class VTag():

    def __init__(self, n=1, tags=10, h5=None):
        self.ARGS = dict(
            n_frames = -1,
            w        = -1,
            h        = -1,
            c        = -1,
            n_id     = n,
            n_tags   = tags,
        )
        self.DATA = dict(
            imgs = None,
            poi  = None
        )

    def load(self, path=".", n=None, bounds=[]):
        """
        parameters
        ---
        n: the first-n files will be loaded

        outputs
        ___

        """
        # list files
        ls_imgs = os.listdir(path)
        ls_imgs.sort()
        files   = [os.path.join(path, f) for f in ls_imgs if ".png" in f]

        # check dimensions
        h, w, c = cv.imread(files[0]).shape

        # load files into `imgs_rgb`
        if n is None: n = len(files)
        imgs = np.zeros((n, h, w), dtype=np.uint8)
        for i in range(n):
            imgs[i] = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
        self.DATA["imgs"] = imgs

        # update contents
        self.ARGS["n_frames"] = n
        self.ARGS["h"]   = h
        self.ARGS["w"]   = w
        self.ARGS["c"]   = c
        self.DATA["poi"] = n * [None]

    def detect_motion(self, n_ticks=2):
        ''''
        use high-motion pixels as pixel of interest (POI)
        '''
        imgs        = self.DATA["imgs"]
        n           = self.ARGS["n_frames"]
        # motion intensity
        imgs_motion = np.zeros(imgs.shape, dtype=np.float32)
        # binary value, poi (1) or not (0)
        imgs_poi    = np.zeros(imgs.shape, dtype=int)

        # compute motion
        for i in range(n):
            imgs_motion[i] = detect_motion(imgs, i)

        # binarize motion image
        cutoff, tick = get_threshold_motion(imgs_motion, n_ticks=n_ticks)
        for i in range(n):
            imgs_poi[i] = get_binary(imgs_motion[i], cutabs=cutoff)

        # increase poi for those frame with low motion
        rescue_low_motion_frame(imgs_poi, imgs_motion, cutoff, tick)

        # bridge frame with no motion detected
        add_vision_persistence(imgs_poi)

        # only keep edges of the deteced motion
        imgs_poi_e = detect_edges(imgs_poi)

        # extract POI(y, x)
        self.DATA["poi"] = get_nonzero_from_img(imgs_poi_e)

    def detect_edges(self, n_denoise=10):
        '''
        '''
        imgs_mov = self.IMGS["mov"]
        # binary of movements, n*h*w
        imgs_edg = self.IMGS["edg"]
        n        = self.ARGS["n"]
        pos_yx   = self.OUTS["pos_yx"]
        bounds   = self.ARGS["bounds"]

        k_edge = np.array((
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]),
            dtype='int')
        k_gauss = np.array((
            [1, 4, 1],
            [4, 9, 4],
            [1, 4, 1]),
            dtype='int') / 29

        for i in range(n):

            conv = convolve2d(imgs_mov[i], k_gauss, mode="same")
            for _ in range(n_denoise):
                conv = convolve2d(conv, k_gauss, mode="same")
            # denoise the movement pictures

            conv = get_binary(conv, cutabs=.5)
            conv = convolve2d(conv, k_edge, mode="same")
            # perform edge detection 

            conv = get_binary(conv, cutabs=.5)
            imgs_edg[i] = conv

            # find pos of edges and filter edges by safe area (boundary)
            pos_yx_tmp = find_nonzeros(imgs_edg[i])
            # get locations and/or numeric color of non-zero pixels
            # number of non-zero pixels * 3
            pos_yx[i]  = filter_edges(pos_yx_tmp, bounds)
            # the location of non-zero pixels within the bounds


    def save(self, h5="model.h5"):
        pickle.dump((self.ARGS, self.IMGS, self.OUTS), open(h5, "wb"))

    def save_labels(self, file="labels.csv"):
        labels = self.OUTS["pred_labels"]
        n_ids  = self.ARGS["n_id"]
        save_labels(labels, n_ids, file)


def detect_motion(imgs, frame, span=1):
    """
    parameters
    ---
    imgs : black and white pictures, n*h*w
    frame: index of frame/picture

    return
    ___
    motion itensity in the same dimensions
    """
    i = frame
    j = span
    # stack 2*2 iamges: out_img is a 4-frame std image
    out_std = []
    for _ in range(2):
    # iterate twice, 2 things in each iteration
        out_std += [
            imgs[i:  (i+j+1)].std(axis=(0, )),
            imgs[(i-j):(i+1)].std(axis=(0, ))
        ]
        j += 1
    # first iteration:
    # sd of every pixel with itself in the next frame, and
    # sd of every pixel with itself in the previous frame
    # second iteration:
    # sd of every pixel with itself in the next two frames, and
    # sd of every pixel with itself in the previous two frames

    out_img = sum(out_std) / len(out_std)
    # average over the 4 things
    return out_img

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
    signals = np.array(signals, dtype=np.float32)
    if cutabs is None:
        # set color to 1 if over 50 percentile
        _, out = cv.threshold(signals,
                        np.quantile(signals, cut), upbound, cv.THRESH_BINARY)
    else:
        # set color to 1 if over cutabs
        _, out = cv.threshold(signals,
                        cutabs, upbound, cv.THRESH_BINARY)
    return out

def get_threshold_motion(imgs_motion, n_ticks=3):
    """
    calculate threshold to filter high-motion pixel

    return
    ---
    cutoff : value to fileter POI
    tick   : s.d. of motion distribution
    """
    nonna_frames = imgs_motion[~np.isnan(imgs_motion).max(axis=(1, 2))]
    tick   = np.std(nonna_frames)
    # median + n_ticks=3 SD (99.7%)
    cutoff = np.median(nonna_frames) + (n_ticks * tick)
    return cutoff, tick

def rescue_low_motion_frame(imgs_poi, imgs_motion, cutoff, tick, rate_rescue=.3):
    """
    inputs
    ---
    cutoff, tick: computed from `get_threshold_motion()`

    return
    ---
    NULL, update imgs_poi
    """
    # count the number of nonzero elements in each frame [n-length vector]
    nsig_frames = np.array([np.count_nonzero(img) for img in imgs_poi])
    # set up threshold for rescuing frame with motion below rate_rescue (0.3) quantile
    cut_rescue  = np.quantile(nsig_frames, rate_rescue)
    # extract the index of frames where nonzero elements are below the threshold
    idx_rescue  = np.where((nsig_frames < cut_rescue) & (nsig_frames > 0))[0]
    for i in idx_rescue:
        adjust  = 0
        # keep looping if the number of nonzero elements is still below threshold
        while np.count_nonzero(imgs_poi[i]) <= cut_rescue:
            adjust += (tick * 0.2)
            # decrease the threshold a little bit when changing into black and white
            imgs_poi[i] = get_binary(imgs_motion[i], cutabs=cutoff - adjust)

def add_vision_persistence(imgs_poi):
    # add the nonzero pixels (value=1) from the next frame to the current one
    imgs_poi[:-1] += (imgs_poi[1:] == 1)
    # add the nonzero pixels (value=1) from the previous frame to the current one
    imgs_poi[1:]  += (imgs_poi[:-1] == 1)

    max_value = 1 * 2 + 1
    cut       = max_value * 0.3 # [1, 0, 0] -> [1, 1, 0] one side
    # change pictures into binary again
    for i in range(len(imgs_poi)):
        imgs_poi[i] = get_binary(imgs_poi[i], cutabs=cut)


def detect_edges(imgs_poi):
    n = len(imgs_poi)
    k_edge = np.array((
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]),
        dtype='int')
    k_gauss = np.array((
        [1, 4, 1],
        [4, 9, 4],
        [1, 4, 1]),
        dtype='int') / 29

    imgs_poi_e = np.zeros(imgs_poi.shape)
    for i in range(n):
        # denoise the movement pictures
        conv = convolve2d(imgs_poi[i], k_gauss, mode="same")
        conv = get_binary(conv, cutabs=.5)
        # perform edge detection
        conv = convolve2d(conv, k_edge, mode="same")
        conv = get_binary(conv, cutabs=.5)
        imgs_poi_e[i] = conv

    return imgs_poi_e


def find_nonzeros(img, is_values=False):
    # img: h*w
    y, x = np.nonzero(img)
    # coordiantes of non-zero pixels
    # length of x, y would be the number of non-zero pixels
    if is_values:
        return np.array([[y[i], x[i], img[y[i], x[i]]] for i in range(len(x))])
        # get locations and the numeric value of non-zero pixels
        # return 3 values for each non-zero pixel
    else:
        return np.array([[y[i], x[i]] for i in range(len(x))])
        # only get the locations of non-zero pixels
        # return 2 values


def get_nonzero_from_img(img):
    n = len(img)
    f, y, x = np.nonzero(img)
    poi = n * [None]
    for i in range(n):
        idx = [f == i]
        poi[i] = [[yi, xi] for yi, xi in zip(y[tuple(idx)], x[tuple(idx)])]
    return poi