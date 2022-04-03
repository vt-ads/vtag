# imports
import os
import numpy  as np
import pandas as pd
import cv2    as cv
import h5py
from matplotlib import pyplot as plt

# vtag functions
from .motion  import detect_motion,\
                     get_threshold_motion,\
                     get_binary,\
                     rescue_low_motion_frame,\
                     add_vision_persistence
from .contour import detect_contour
from .utils   import get_nonzero_from_img
from .tracking import track, cluster_poi

class VTag():

    def __init__(self, k=3):
        self.ARGS = dict(
            n   = -1,
            w   = -1,
            h   = -1,
            c   = -1,
            k   = k
        )
        self.DATA = dict(
            imgs  = None,
            poi   = None,
            lbs   = None,
            error = None,
        )
        self.path = ""
        self.has_h5 = False

    def load(self, path=".", n=None, h5=None):
        """
        parameters
        ---
        n: the first-n files will be loaded
        """
        # list files
        self.path = path
        ls_files = os.listdir(path)
        ls_files.sort()
        ls_imgs  = [os.path.join(path, f) for f in ls_files if ".png" in f]
        h5_file  = [os.path.join(path, f) for f in ls_files if "vtag.h5" in f]

        # check dimensions
        h, w, c = cv.imread(ls_imgs[0]).shape

        # load files into `imgs`
        if n is None: n = len(ls_imgs)
        imgs = np.zeros((n, h, w), dtype=np.uint8)
        for i in range(n):
            imgs[i, :, :] = cv.imread(ls_imgs[i], cv.IMREAD_GRAYSCALE)
        self.DATA["imgs"] = imgs

        # update data
        self.ARGS["n"] = n
        self.ARGS["h"] = h
        self.ARGS["w"] = w
        self.ARGS["c"] = c
        # check if h5 exists
        if isinstance(h5, str):
            h5_file += [h5]

        if len(h5_file) != 0:
            with h5py.File(h5_file[-1], "r") as f:
                print("Loaded: %s" % h5_file[-1])
                self.has_h5 = True
                self.DATA["lbs"]   = f["lbs"][:]
                self.DATA["error"] = f["error"][:]
                self.DATA["poi"]   = pd.DataFrame(f["poi"][:]) # cuz it's numpy in h5
                self.DATA["poi"].columns = ["frame", "y", "x"]
                self.ARGS["k"]     = self.DATA["lbs"].shape[1]
        else:
            self.DATA["error"] = np.zeros((self.ARGS["n"], self.ARGS["k"]))
            self.DATA["lbs"]   = np.zeros((self.ARGS["n"], self.ARGS["k"], 2))

    def detect_poi(self, n_ticks=2, n_denoise=2):
        ''''
        detect high-motion pixels as pixel of interest (POI)
        '''
        imgs = self.DATA["imgs"]
        n    = self.ARGS["n"]
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
        imgs_poi_e = detect_contour(imgs_poi, n_denoise=n_denoise)

        # extract POI(y, x)
        self.DATA["poi"] = get_nonzero_from_img(imgs_poi_e)

    def track(self, frame, tracker, pts_init=None, winSize=(70, 70)):
        """
        parameters
        ---
        frame   : integer, which frame to start tracking
        pts_init: k by 2 (xy) numpy matrix
        """
        ls_err = self.DATA["error"]
        ls_pts = self.DATA["lbs"]

        # init points
        if pts_init is None:
            lbs, pts_init = cluster_poi(self.poi(frame), self.ARGS["k"],
                                        method="agglo")
        ls_pts[frame] = pts_init

        # define parameters
        kwargs = dict({"imgs": self.DATA["imgs"],
                       "idx": frame,
                       "tracker": tracker,
                       "pts_idx": pts_init,
                       "winSize": winSize})

        # forward (i=10, st=11, ed=20: [11,12,...,18,19])
        st, ed = frame + 1, self.ARGS["n"]
        ls_pts[st:], ls_err[st:] = track(st=st, ed=ed, **kwargs)
        # # backward (i=10, st=9, ed=0: [9,8,7,6,5,4,3,2,1])
        # st, ed = frame - 1, 0
        # ls_pts[1:frame], ls_err[1:frame] = track(st=st, ed=ed, **kwargs)

    def save(self, filename="vtag.h5"):
        # reduce size of poi dataframe
        cols = self.DATA["poi"].columns
        self.DATA["poi"][cols] = self.DATA["poi"][cols].apply(pd.to_numeric,
                                                      downcast="unsigned")
        # use hdf5 to compress results
        out_h5 = os.path.join(self.path, filename)
        with h5py.File(out_h5, "w") as f:
            param = {"compression": "gzip", "chunks": True}
            f.create_dataset("lbs",   data=self.DATA["lbs"],   **param)
            f.create_dataset("error", data=self.DATA["error"], **param)
            f.create_dataset("poi",   data=self.DATA["poi"],   **param)

    def evaluate(self, truth):
        return np.mean(np.square(self.DATA["lbs"] - truth).sum(axis=2)**.5, axis=1)

    # getters---
    def poi(self, frame):
        if self.DATA["poi"] is not None:
            return self.DATA["poi"].query("frame == %d" % frame)[["x", "y"]]
        else:
            return -1

    def img(self, frame):
        if self.DATA["imgs"] is not None:
            return self.DATA["imgs"][frame]
        else:
            return -1

    def lbs(self, frame):
        if self.DATA["lbs"] is not None:
            return self.DATA["lbs"][frame]
        else:
            return -1

    def error(self, frame):
        if self.DATA["error"] is not None:
            return self.DATA["error"][frame]
        else:
            return -1

    def mask(self, frame):
        img = self.img(frame)
        poi = self.poi(frame)
        # iterate each poi
        img_mask = np.zeros(img.shape, dtype=int)
        for _, item in poi.iterrows():
            img_mask[item.y, item.x] = 1
        # output mask matrix (2D binary matrix)
        return img_mask

    def snapshot(self, frame):
        img = self.img(frame)
        lbs = self.lbs(frame)
        plt.imshow(img)
        plt.scatter(lbs[:, 0], lbs[:, 1], marker="x", c='red', s=50)

    # GUI---
    def update_k(self, k):
        """
        only use when re-calculation is needed
        """
        self.ARGS["k"] = k
        self.DATA["error"] = np.zeros((self.ARGS["n"], self.ARGS["k"]))
        self.DATA["lbs"]   = np.zeros((self.ARGS["n"], self.ARGS["k"], 2))

