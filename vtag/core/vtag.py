# imports
import os
import numpy  as np
import pandas as pd
import cv2    as cv
import h5py
from matplotlib import pyplot as plt
from PIL import Image

# vtag functions
from .vtstream import VTStream
from .motion  import detect_motion,\
                     get_threshold_motion,\
                     get_binary,\
                     rescue_low_motion_frame,\
                     add_vision_persistence,\
                     get_nonzero_from_img
from .clustering import cluster_poi, sort_points
from .contour  import detect_contour, get_mask, bbox_img
from .tracking import track

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
            poi   = None,
            lbs   = None,
            error = None,
            area  = None,
        )
        self.stream = VTStream()

    def load(self, path="."):
        """
        parameters
        ---
        n: the first-n files will be loaded
        """
        self.stream.load(path)
        self.ARGS["n"], self.ARGS["h"], self.ARGS["w"] = self.stream.get_meta()
        self.DATA["poi"]   = [np.array([], dtype=int) for _ in range(self.ARGS["n"])] # xyk
        self.DATA["lbs"]   = np.zeros((self.ARGS["n"], self.ARGS["k"], 2))
        self.DATA["area"]  = np.zeros((self.ARGS["n"], self.ARGS["k"]))
        self.DATA["error"] = np.ones((self.ARGS["n"],  self.ARGS["k"])) * 999

    def load_h5(self, h5):
        # check if h5 exists
        h5_file  = [os.path.join(path, f) for f in os.listdir(self.stream.dirname) if "vtag.h5" in f]
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
        pass

    def detect_poi(self, frame, window=30, n_ticks=2, n_denoise=2):
        ''''
        detect high-motion pixels as pixel of interest (POI)
        '''
        # check if already had poi defined
        for f in range(frame, frame + window):
            if len(self.DATA["poi"][f]) != 0:
                frame  += 1
                window -= 1
            else:
                break
        if window == 0: return 0

        # load images
        imgs = self.img(frame, frame + window)

        # motion intensity
        imgs_motion = np.zeros(imgs.shape, dtype=np.float32)
        # binary value, poi (1) or not (0)
        imgs_poi    = np.zeros(imgs.shape, dtype=int)

        # compute motion
        for i in range(window):
            imgs_motion[i] = detect_motion(imgs, i)

        # binarize motion image
        cutoff, tick = get_threshold_motion(imgs_motion, n_ticks=n_ticks)
        for i in range(window):
            imgs_poi[i] = get_binary(imgs_motion[i], cutabs=cutoff)

        # increase poi for those frame with low motion
        rescue_low_motion_frame(imgs_poi, imgs_motion, cutoff, tick)

        # bridge frame with no motion detected
        add_vision_persistence(imgs_poi)

        # only keep edges of the deteced motion
        imgs_poi_e = detect_contour(imgs_poi, n_denoise=n_denoise)

        # extract POI (x, y)
        self.DATA["poi"][frame : frame + window] = get_nonzero_from_img(imgs_poi_e)
        for f in range(frame, frame + window):
            self.cluster(f)

    def cluster(self, frame):
        """
        cluster poi and update DATA["lbs"]
        return centroids of each cluster
        """
        # if has no poi defined, skip
        poi = self.poi(frame)
        if len(poi) == 0:
            return 0
        # cluster pois
        clusters, cts = cluster_poi(poi, self.ARGS["k"], method="cv")
        # clusters, cts = cluster_poi(poi, self.ARGS["k"], method="agglo")

        # update labels
        lbs = self.lbs(frame)
        if np.sum(lbs) == 0:
            # if labels are never defined, use cnetroids (cts) as new labels
            self.DATA["lbs"][frame] = cts
            sorted_clusters = clusters
        else:
            # otherwise, reassign (sort) cluster number
            cts, order = sort_points(cts, lbs)
            sorted_clusters = clusters.copy()
            for i in range(self.ARGS["k"]):
                sorted_clusters[clusters == i] = order[i]

        # update poi cluster
        self.DATA["poi"][frame][:, 2] = sorted_clusters

        # output
        return cts

    def track(self, frame, tracker="SparseLK", pts_init=None, win_xy=(70, 70)):
        """
        parameters
        ---
        frame   : integer, which frame to start tracking
        pts_init: k by 2 (xy) numpy matrix
        """
        ls_err = self.DATA["error"]
        ls_pts = self.DATA["lbs"]
        win_t = 100
        # init points
        if pts_init is None:
            if len(self.poik(frame)) == 0:
                self.detect_poi(frame=frame)
            pts_init = self.cluster(frame)

        # define parameters
        # kwargs = dict({"imgs": self.DATA["imgs"],
        st = frame
        ed = frame + win_t
        kwargs = dict({"imgs": self.stream.get(st, ed),
                       "pts_0": np.array(pts_init),
                       "win_xy" : win_xy})

        # forward (i=10, st=11, ed=20: [11,12,...,18,19])
        # st, ed = frame + 1, self.ARGS["n"]
        ls_pts[st:ed], ls_err[st:ed] = track(tracker=tracker, **kwargs)
        # # backward (i=10, st=9, ed=0: [9,8,7,6,5,4,3,2,1])
        # st, ed = frame - 1, 0
        # ls_pts[1:frame], ls_err[1:frame] = track(st=st, ed=ed, **kwargs)

    def save_mask(self, frame):
        """
        update area and save mask images
        """
        for k in range(self.ARGS["k"]):
            poi  = self.poi(frame=frame, k=k).reshape((-1, 1, 2))
            img  = self.img(frame)
            bbox = cv.boundingRect(poi)

            # mask
            img_mask = get_mask(img, bbox, poi)
            self.DATA["area"][frame, k] = np.sum(img_mask)

            # original image
            img_img = bbox_img(img, bbox)

            # output path
            dirsave = self.stream.dirsave
            dirmask = os.path.join(dirsave, "mask")
            if not os.path.exists(dirmask):
                os.makedirs(dirmask)

            # export
            path_img  = os.path.join(dirmask, "id%d_f%d_image.jpg" % (k, frame))
            path_mask = os.path.join(dirmask, "id%d_f%d_mask.jpg"  % (k, frame))
            path_seg  = os.path.join(dirmask, "id%d_f%d_seg.jpg"   % (k, frame))
            Image.fromarray(img_img).save(path_img)
            Image.fromarray(img_mask).save(path_mask)
            Image.fromarray(img_img * img_mask).save(path_seg)

    def save(self, filename="vtag.h5"):
        # use hdf5 to compress results
        out_h5 = os.path.join(self.stream.dirsave, filename)
        with h5py.File(out_h5, "w") as f:
            param = {"compression": "gzip", "chunks": True}
            f.create_dataset("n",     data=self.ARGS["n"],   **param)
            f.create_dataset("w",     data=self.ARGS["w"],   **param)
            f.create_dataset("h",     data=self.ARGS["h"],   **param)
            f.create_dataset("k",     data=self.ARGS["k"],   **param)
            f.create_dataset("poi",   data=self.DATA["poi"],   **param)
            f.create_dataset("lbs",   data=self.DATA["lbs"],   **param)
            f.create_dataset("error", data=self.DATA["error"], **param)
            f.create_dataset("area",  data=self.DATA["poi"],   **param)
        # reduce size of poi dataframe
        # cols = self.DATA["poi"].columns
        # self.DATA["poi"][cols] = self.DATA["poi"][cols].apply(pd.to_numeric,
        #                                               downcast="unsigned")

    def evaluate(self, truth):
        return np.mean(np.square(self.DATA["lbs"] - truth).sum(axis=2)**.5, axis=1)

    # getters---
    def poi(self, frame, k=None):
        poi = self.DATA["poi"][frame]
        if k is None:
            # return all poi(xy)
            return poi[:, :2] # xy
        else:
            # return only poi(xy) labeled as k
            idx = poi[:, 2] == k
            return poi[idx][:, :2]

    def poik(self, frame):
        return self.DATA["poi"][frame] # xyk

    def img(self, frame, frame_end=None):
        if not self.stream.isEmpty:
            if frame_end is None:
                return self.stream.get(frame)
            else:
                return self.stream.get(frame, frame_end)
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
        poi = self.poik(frame)
        # iterate each poi
        img_mask = np.zeros(img.shape, dtype=int)
        for x, y, k in poi:
            # background: 0, k0: 1, k1: 2, ...
            img_mask[y, x] = k + 1
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

