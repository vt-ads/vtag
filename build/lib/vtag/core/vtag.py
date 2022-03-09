# imports
import os
import numpy  as np
import cv2    as cv

# vtag functions
from .motion  import detect_motion,\
                     get_threshold_motion,\
                     get_binary,\
                     rescue_low_motion_frame,\
                     add_vision_persistence
from .contour import detect_contour
from .utils   import get_nonzero_from_img
from .tracking import LK_tracking, cluster_poi

class VTag():

    def __init__(self, k=1, tags=10, h5=None):
        self.ARGS = dict(
            n        = -1,
            w        = -1,
            h        = -1,
            c        = -1,
            k        = k,
            n_tags   = tags,
        )
        self.DATA = dict(
            imgs  = None,
            poi   = None,
            error = None,
            track = None,
        )

    def load(self, path=".", n=None, bounds=[]):
        """
        parameters
        ---
        n: the first-n files will be loaded
        """
        # list files
        ls_imgs = os.listdir(path)
        ls_imgs.sort()
        files   = [os.path.join(path, f) for f in ls_imgs if ".png" in f]

        # check dimensions
        h, w, c = cv.imread(files[0]).shape

        # load files into `imgs`
        if n is None: n = len(files)
        imgs = np.zeros((n, h, w), dtype=np.uint8)
        for i in range(n):
            imgs[i] = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
        self.DATA["imgs"] = imgs

        # update data
        self.ARGS["n"] = n
        self.ARGS["h"] = h
        self.ARGS["w"] = w
        self.ARGS["c"] = c
        self.DATA["error"] = np.zeros((self.ARGS["n"], self.ARGS["k"]))
        self.DATA["track"] = np.zeros((self.ARGS["n"], self.ARGS["k"], 2))

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

    def track(self, frame):
        ls_err = self.DATA["error"]
        ls_pts = self.DATA["track"]

        # init points
        lbs, pts_init = cluster_poi(self.poi(frame), self.ARGS["k"],
                                    method="agglo")

        # forward (i=10, st=11, ed=20: [11,12,...,18,19])
        st, ed = frame + 1, self.ARGS["n"]
        ls_pts[st:], ls_err[st:] = LK_tracking(self.DATA["imgs"],
                                        idx=frame, pts_idx=pts_init,
                                        st=st, ed=ed)
        # backward (i=10, st=9, ed=0: [9,8,7,6,5,4,3,2,1])
        st, ed = frame - 1, 0
        ls_pts[1:frame], ls_err[1:frame] = LK_tracking(self.DATA["imgs"],
                                                idx=frame, pts_idx=pts_init,
                                                st=st, ed=ed)
    # getters---
    def poi(self, frame):
        return self.DATA["poi"].query("frame == %d" % frame)[["x", "y"]]

    def img(self, frame):
        return self.DATA["imgs"][frame]