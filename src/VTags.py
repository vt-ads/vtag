from lib import *

class VTags():

    def __init__(self, k=1, n_tags=10, h5=None):
        if h5 is None:
            self.ARGS = dict(
                            n      = -1,
                            w      = -1,
                            h      = -1,
                            c      = -1,
                            n_id   = k,
                            n_tags = n_tags,
                            k      = k * n_tags,
                            bounds = [],
            )
            # all IMGS is (n, h, w, -1)
            self.IMGS = dict(
                            rgb = None,
                            bw  = None,
                            mov = None,
                            edg = None,
                            pred= None,
            )
            self.OUTS = dict(
                            features = None, # (y, x) coordinate of centers of each k
                            pos_yx   = None,  # (y, x) coordinate of edges
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
        files   = [f for f in ls_imgs if ".png" in f]

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
        if len(bounds) == 0:
            # [y, x]
            self.ARGS["bounds"] = np.array([[0, 0], [0, w], [h, w], [h, 0]])
        else:
            self.ARGS["bounds"] = np.array(bounds)

        self.IMGS["rgb"]      = imgs_rgb
        self.IMGS["bw"]       = imgs_rgb.sum(axis=3)
        self.IMGS["mov"]      = np.zeros((n, h, w), dtype=np.float32)
        self.IMGS["edg"]      = np.zeros((n, h, w), dtype=np.uint8)
        self.IMGS["pred"]     = np.zeros((n, h, w), dtype=np.uint8)
        self.IMGS["pred_cls"] = np.zeros((n, h, w), dtype=np.uint8)

        self.OUTS["pos_yx"]      = n * [None]
        self.OUTS["features"]    = np.zeros((n, self.ARGS["k"], 24))
        self.OUTS["k_to_id"]     = np.zeros((n, self.ARGS["k"]))
        self.OUTS["pcs"]         = np.zeros((n, self.ARGS["k"], 2))
        self.OUTS["pred_labels"] = np.zeros((n, self.ARGS["n_id"], 2), dtype=np.int)
        # not used
        self.OUTS["cls"]         = n * [None]

        self.OUTS["labels"] = load_labels(self.ARGS["n"], self.ARGS["n_id"])

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
        imgs_bw  = self.IMGS["bw"]
        n        = self.ARGS["n"]
        imgs_mov_tmp = imgs_mov.copy()

        # compute std images
        for i in range(n):
            imgs_mov_tmp[i] = detect_imgs(imgs_bw, i)

        # binarize
        # remove na frames
        nonna_frames = imgs_mov_tmp[~np.isnan(imgs_mov_tmp).max(axis=(1, 2))]
        tick   = np.std(nonna_frames)
        cutoff = np.median(nonna_frames) + (n_ticks * tick)
        for i in range(n):
            imgs_mov[i] = get_binary(imgs_mov_tmp[i], cutabs=cutoff)

        # rescue low-signal frame
        nsig_frames = np.array([np.count_nonzero(img) for img in imgs_mov])
        cut_rescue  = np.quantile(nsig_frames, .3)
        idx_rescue  = np.where((nsig_frames < cut_rescue) & (nsig_frames > 0))[0]
        for i in idx_rescue:
            img_tmp = imgs_mov[i]
            adjust  = 0
            while np.count_nonzero(img_tmp) <= cut_rescue:
                adjust += (tick * 0.2)
                img_tmp = get_binary(imgs_mov_tmp[i], cutabs=cutoff - adjust)
            imgs_mov[i] = img_tmp

        # Persistence of vision
        for i in range(n_blend):
            idx = i + 1
            imgs_mov[:-idx] += (imgs_mov[idx:] > 0)
            imgs_mov[idx:]  += (imgs_mov[:-idx] > 0)
        max_value = n_blend * 2 + 1
        imgs_mov  = imgs_mov / max_value
        cut       = max_value * 0.5
        imgs_mov  = get_binary(imgs_mov, cutabs=cut)


    def detect_edges(self, n_denoise=10):
        '''
        '''
        imgs_mov = self.IMGS["mov"]
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
            # # Strategy I
            # conv = convolve2d(imgs_mov[i], k_edge, mode="same")
            # conv = get_binary(conv)
            # for _ in range(n_denoise):
            #     conv = convolve2d(conv, k_gauss, mode="same")
            #     conv = get_binary(conv, cutabs=.5)

            # Strategy II
            conv = convolve2d(imgs_mov[i], k_gauss, mode="same")
            for _ in range(n_denoise):
                conv = convolve2d(conv, k_gauss, mode="same")
            conv = get_binary(conv, cutabs=.5)
            conv = convolve2d(conv, k_edge, mode="same")
            conv = get_binary(conv, cutabs=.5)
            imgs_edg[i] = conv
            # create fake signals to avoid empty array error
            imgs_edg[i, 5:10, 5:10] = 1
            # find pos of edges and filter edges by safe area (boundary)
            pos_yx_tmp = find_nonzeros(imgs_edg[i])
            pos_yx[i]  = filter_edges(pos_yx_tmp, bounds)

    def detect_clusters(self):
        '''
        '''
        n        = self.ARGS["n"]
        k        = self.ARGS["k"]
        imgs_edg = self.IMGS["edg"]
        clusters = self.OUTS["cls"]
        pos_yx   = self.OUTS["pos_yx"]
        # 24 is: time(8)+spatial(4)+xy(12))
        centers  = np.zeros((n, k, 24))

        for i in range(n):
            try:
                cls, cts, np_yx = do_k_means(imgs_edg, pos_yx[i], i, k)
                clusters[i] = cls
                centers[i]  = cts
                pos_yx[i]   = np_yx
            except Exception as e:
                print(e)

        self.OUTS["features"] = centers

    def map_k_to_id(self):
        '''
        '''
        n            = self.ARGS["n"]
        k            = self.ARGS["n_id"]
        features_all = self.OUTS["features"]

        for i in range(n):
            self.OUTS["k_to_id"][i], self.OUTS["pcs"][i] =\
                map_features_to_id(features_all[i], k)

    def make_predictions(self):
        '''
        '''
        n        = self.ARGS["n"]
        k        = self.ARGS["n_id"]
        clts     = self.OUTS["cls"]
        pos_yx   = self.OUTS["pos_yx"]
        k_to_id  = self.OUTS["k_to_id"]
        pred     = self.IMGS["pred"]
        pred_clt = self.IMGS["pred_cls"]

        for i in range(n):
            if clts[i] is not None:
                clt = clts[i]
                pts = pos_yx[i].astype(np.int)

                # show prediction
                which_id = k_to_id[i][clt]
                pred[i][pts[:, 0], pts[:, 1]] = which_id

                # show clusters
                pred_clt[i][pts[:, 0], pts[:, 1]] = clt + 1 # cluster from 1 to k

        # refine predictions
        # n, h, w = pred.shape
        # n_blend = 2
        # max_value = n_blend + 1
        # new_pred = np.zeros((n, h, w, k))
        # for i in range(k):
        #     new_pred[pred == (i + 1), i] = 1
        #     for j in range(n_blend):
        #         idx = j + 1
        #         # new_pred[:-idx, :, :, i] += new_pred[idx:, :, :, i]
        #         new_pred[idx:, :, :, i] += (new_pred[:-idx, :, :, i] > 0)
        #     new_pred[:, :, :, i] = new_pred[:, :, :, i] / max_value
        # # idx0 = np.sum(new_pred, axis=3) == 0
        # idx1 = new_pred[:, :, :, 0] > new_pred[:, :, :, 1]
        # idx2 = new_pred[:, :, :, 0] < new_pred[:, :, :, 1]
        # pred[idx1] = 1
        # pred[idx2] = 2

    def create_labels(self):
        pred     = self.IMGS["pred"]
        clusters = make_labels(pred)
        self.OUTS["pred_labels"] = sort_clusters(clusters, pred)

    def set_labels(self, i, x, y):
        """
        x, y: array of coordinates of x or y
        """
        new_labels = np.array([x, y]).swapaxes(0, 1)[:, [1, 0]] # makes it (y, x)
        self.OUTS["labels"][i] = new_labels

    def save(self, h5="model.h5"):
        pickle.dump((self.ARGS, self.IMGS, self.OUTS), open(h5, "wb"))

    def save_labels(self, file="labels.csv"):
        labels = self.OUTS["pred_labels"]
        n_ids  = self.ARGS["n_id"]
        save_labels(labels, n_ids, file)


