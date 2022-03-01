# import pickle
# import os
# import numpy as np
# import cv2   as cv
# from .lib import get_binary

class VTag():

    def __init__(self, n=1, tags=10, h5=None):
        if h5 is None:
            self.ARG = dict(
                n_frames = -1,
                w        = -1,
                h        = -1,
                c        = -1,
                n_id     = n,
                n_tags   = tags,
                bounds = [],
            )
            self.DATA = dict(
                            rgb = None,
                            bw  = None,
                            mov = None,
                            edg = None,
                            pred= None,
            )
            self.OUTS = dict(
                            features = None, # (y, x) coordinate of centers of each k
                            pos_yx   = None, # (y, x) coordinate of edges
                            k_to_id  = None,
                            pcs = None,
                            pred_cls    = None,
                            pred_labels = None,
                            labels = None,
                            # not used
                            cls    = None   # clusters
            )
        else:
            self.ARGS, self.IMGS, self.OUTS = pickle.load(open(h5, "rb"))

    def load(self, path=".", n=-1, bounds=[]):
        # list files
        ls_imgs = os.listdir(path)
        ls_imgs.sort()
        files   = [os.path.join(path, f) for f in ls_imgs if ".png" in f]

        # check dimensions
        h, w, c = cv.imread(files[0]).shape

        # create np matrix
        if n == -1:
            n = len(files)
        imgs_rgb = np.zeros((n, h, w, c), dtype=np.uint8)

        # iterate through files
        for i in range(n):
            imgs_rgb[i] = cv.imread(files[i])

        # update contents
        self.ARGS["n"] = n
        self.ARGS["h"] = h
        self.ARGS["w"] = w
        self.ARGS["c"] = c
        # fill in the dimension of pictures
        if len(bounds) == 0:
            # [y, x]
            self.ARGS["bounds"] = np.array([[0, 0], [0, w], [h, w], [h, 0]])
        else:
            self.ARGS["bounds"] = np.array(bounds)
        # "bounds" is the range on the pictures VTag should consider

        self.IMGS["rgb"]      = imgs_rgb
        # original pictures
        # n*h*w*c
        self.IMGS["bw"]       = imgs_rgb.sum(axis=3)
        # convert pictures into black and white
        # n*h*w
        self.IMGS["mov"]      = np.zeros((n, h, w), dtype=np.float32)
        self.IMGS["edg"]      = np.zeros((n, h, w), dtype=np.uint8)
        self.IMGS["pred"]     = np.zeros((n, h, w), dtype=np.uint8)
        self.IMGS["pred_cls"] = np.zeros((n, h, w), dtype=np.uint8)

        self.OUTS["pos_yx"]      = n * [None]
        self.OUTS["features"]    = np.zeros((n, self.ARGS["k"], 24))
        self.OUTS["k_to_id"]     = np.zeros((n, self.ARGS["k"]))
        self.OUTS["pcs"]         = np.zeros((n, self.ARGS["k"], 2))
        self.OUTS["pred_labels"] = np.zeros((n, self.ARGS["n_id"], 2), dtype=np.int)
        # k: total number of clusters/sub-groups
        # n_id: number of animals
        # not used
        self.OUTS["cls"]         = n * [None]
        self.OUTS["labels"] = load_labels(self.ARGS["n"], self.ARGS["n_id"])
        # see "load_labels" in "lib"

    def run(self):
        self.detect_movements()
        self.detect_edges()
        self.detect_clusters()
        self.map_k_to_id()
        self.make_predictions()
        self.create_labels()

    def detect_movements(self, n_ticks=3, n_blend=1):
        '''
        '''
        imgs_mov = self.IMGS["mov"]
        # n*h*w
        imgs_bw  = self.IMGS["bw"]
        # n*h*w
        n        = self.ARGS["n"]
        # number of frames/pictures
        imgs_mov_tmp = imgs_mov.copy()

        # compute std images
        for i in range(n):
            imgs_mov_tmp[i] = detect_imgs(imgs_bw, i)
        # see "detect_imgs" in "lib"
        # n*h*w
        # binarize
        # remove na frames
        nonna_frames = imgs_mov_tmp[~np.isnan(imgs_mov_tmp).max(axis=(1, 2))]
        # np.isnan => true and false
        # max => true's 
        # ~ => false
        # nonna_frame => extract frames that doesn't contain NA 
        tick   = np.std(nonna_frames)
        cutoff = np.median(nonna_frames) + (n_ticks * tick)
        # constructing some threshold 
        for i in range(n):
            imgs_mov[i] = get_binary(imgs_mov_tmp[i], cutabs=cutoff)
        # change into black and white pictures 

        # rescue low-signal frame
        nsig_frames = np.array([np.count_nonzero(img) for img in imgs_mov])
        # count the number of nonzero elements in each frame
        cut_rescue  = np.quantile(nsig_frames, .3)
        # set up threshold for rescuing as 30 percentile
        idx_rescue  = np.where((nsig_frames < cut_rescue) & (nsig_frames > 0))[0]
        # extract the index of frames where nonzero elements are below the threshold
        for i in idx_rescue:
            img_tmp = imgs_mov[i]
            adjust  = 0
            while np.count_nonzero(img_tmp) <= cut_rescue:
            # keep looping if the number of nonzero elements is below threshold
                adjust += (tick * 0.2)
                img_tmp = get_binary(imgs_mov_tmp[i], cutabs=cutoff - adjust)
                # decrease the threshold a little bit when changing into black and white
            imgs_mov[i] = img_tmp

        # Persistence of vision
        for i in range(n_blend):
        # iterate once 
            idx = i + 1 # idx=1
            imgs_mov[:-idx] += (imgs_mov[idx:] > 0)
            # add the nonzero pixels (value=1) from the next frame to the current one 
            imgs_mov[idx:]  += (imgs_mov[:-idx] > 0)
            # add the nonzero pixels (value=1) from the previous frame to the current one
        max_value = n_blend * 2 + 1
        imgs_mov  = imgs_mov / max_value
        cut       = max_value * 0.5
        imgs_mov  = get_binary(imgs_mov, cutabs=cut)
        # change pictures into binary again

    def save(self, h5="model.h5"):
        pickle.dump((self.ARGS, self.IMGS, self.OUTS), open(h5, "wb"))

    def save_labels(self, file="labels.csv"):
        labels = self.OUTS["pred_labels"]
        n_ids  = self.ARGS["n_id"]
        save_labels(labels, n_ids, file)


