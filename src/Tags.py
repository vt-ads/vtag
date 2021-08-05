from lib import *

class VTags():

    def __init__(self, k=1, n_tags=5):
        self.ARGS = dict(
                        n      = -1,
                        w      = -1,
                        h      = -1,
                        c      = -1,
                        n_id   = k,
                        n_tags = n_tags,
                        k      = k * n_tags,
        )
        self.IMGS = dict(
                        bw  = None,
                        rgb = None,
                        mov = None,
                        edg = None,
        )
        self.OUTS = dict(
                        cts = None,    # (y, x) coordinate of centers of each k
                        cls = None,    # clusters
                        yx_edg = None, # (y, x) coordinate of edges
                        dist   = None, # distance travel to i frame
                        dct    = None, # direction to i frame
                        k_to_id= None,
                        pred = None,
        )

    def load(self, path=".", h5=None, n=-1):
        if h5 is not None:
            self.ARGS, self.IMGS, self.OUTS = pickle.load(open(h5, "rb"))
        else:
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

            self.IMGS["rgb"]    = imgs_rgb
            self.IMGS["bw"]     = imgs_rgb.sum(axis=3)
            self.IMGS["mov"]    = np.zeros((n, h, w), dtype=np.float32)
            self.IMGS["edg"]    = np.zeros((n, h, w), dtype=np.uint8)

            self.OUTS["cts"]    = np.zeros((n, self.ARGS["k"], 14), dtype=np.int) # 14 is: time(8)+spatial(4)+xy(2)
            self.OUTS["cls"]    = n * [None]
            self.OUTS["yx_edg"] = n * [None]
            self.OUTS["dist"]   = np.zeros((n, self.ARGS["k"]))
            self.OUTS["dct"]    = np.zeros((n, self.ARGS["k"], 2))
            self.OUTS["k_to_id"]= np.zeros((self.ARGS["k"]))
            self.OUTS["pred"]   = np.zeros((n, h, w), dtype=np.uint8)


    def run(self):
        self.detect_movements()
        self.detect_edges()
        self.detect_clusters()
        self.sort_clusters()
        self.map_k_to_id()
        self.make_predictions()

    def detect_movements(self):
        imgs_mov = self.IMGS["mov"]
        imgs_bw  = self.IMGS["bw"]
        n        = self.ARGS["n"]

        for i in range(n):
            imgs_mov[i] = detect_imgs(imgs_bw, i)

        # binarize
        std_matrix = imgs_mov[~np.isnan(imgs_mov).max(axis=(1, 2))]
        cutoff = np.median(std_matrix) + (3 * np.std(std_matrix))
        for i in range(n):
            imgs_mov[i] = get_binary(imgs_mov[i], cutabs=cutoff)


    def detect_edges(self, n_denoise=5):
        k_edge = np.array((
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]),
            dtype='int')
        k_gauss = np.array((
            [1, 4, 1],
            [4, 9, 4],
            [1, 4, 1]),
            dtype='int') / 29

        imgs_mov = self.IMGS["mov"]
        imgs_edg = self.IMGS["edg"]
        n        = self.ARGS["n"]

        for i in range(n):
            conv = convolve2d(imgs_mov[i], k_edge, mode="same")
            conv = get_binary(conv)
            for _ in range(n_denoise):
                conv = convolve2d(conv, k_gauss, mode="same")
                conv = get_binary(conv, cutabs=.5)
            imgs_edg[i] = conv

    def detect_clusters(self):
        imgs_edg = self.IMGS["edg"]
        clusters = self.OUTS["cls"]
        centers  = self.OUTS["cts"]
        yx_edges = self.OUTS["yx_edg"]
        k        = self.ARGS["k"]
        n        = self.ARGS["n"]

        for i in range(n):
            try:
                cls, cts, np_yx = do_k_means(imgs_edg, i, k)
                clusters[i] = cls
                centers[i]  = cts
                yx_edges[i] = np_yx
            except Exception as e:
                print(e)


    def sort_clusters(self):
        '''
        NOTE: can be sort by colors and coordinate

        Sort k centers in cts[i + 1] based on distance between (i) and (i + 1)
        args
            i  : ith frame
        Time complexity O = (k-1)^2
        '''
        k = self.ARGS["k"]

        for i in range(self.ARGS["n"] - 1): # the last frame is not applicable
            ls_min = []

            # if k = 3, 0-0, 0-1, 0-2, 1-1, 1-2
            # Decide link k1 (i) to which k2 (i+1)
            for k1 in range(0, k - 1): # frame i, cluster k1
                ls_dist = []

                for k2 in range(k1, k): # frame i + 1, cluster k2
                    ls_dist += [distance(self.OUTS["cts"][i][k1],
                                         self.OUTS["cts"][i + 1][k2])]
                # find min idx
                val_min = np.min(ls_dist)
                idx_min = np.where(ls_dist == val_min)[0][0] + k1
                ls_min += [val_min]

                # swap centers at i+1
                tmp = self.OUTS["cts"][i + 1][k1].copy()
                self.OUTS["cts"][i + 1][k1] = self.OUTS["cts"][i + 1][idx_min]
                self.OUTS["cts"][i + 1][idx_min] = tmp

                # swap clustering i+1 on the position of k1 and idx_min
                if self.OUTS["cls"][i + 1] is not None: # skipped for last frame
                    pos_k1  = self.OUTS["cls"][i + 1] == k1
                    pos_min = self.OUTS["cls"][i + 1] == idx_min
                    self.OUTS["cls"][i + 1][pos_k1] = idx_min
                    self.OUTS["cls"][i + 1][pos_min] = k1

            # compute the distance of the last k
            # so that you will have length k vector for distance between i and i+1 
            ls_min += [distance(self.OUTS["cts"][i][k - 1],
                                self.OUTS["cts"][i + 1][k - 1])] 
            # store distances and direction
            self.OUTS["dist"][i + 1] = ls_min
            self.OUTS["dct"][i + 1]  = self.OUTS["cts"][i + 1] - self.OUTS["cts"][i]

    def map_k_to_id(self):
        '''
        Use moving direction between frames to cluster each sub-cluster
        '''
        fts_dir = np.swapaxes(self.OUTS["dct"], 1, 0).\
            reshape((self.ARGS["k"], -1)).\
            astype(np.float32)

        k_to_id, _ = cv_k_means(fts_dir, self.ARGS["n_id"])
        k_to_id += 1  # which k cluster -> which id
        self.OUTS["k_to_id"] = k_to_id
        # DEBUG === === === === === === === === ===
        self.OUTS["k_to_id"] = np.array(list(range(1, self.ARGS["k"] + 1)))

        # sd = np.std(self.OUTS["dist"])
        # cut = np.median(self.OUTS["dist"]) + 2 * sd
        # # n by k
        # show_k = self.OUTS["dist"] <= cut

    def make_predictions(self):
        clts     = self.OUTS["cls"]
        yx_edges = self.OUTS["yx_edg"]
        pred     = self.OUTS["pred"]
        k_to_id  = self.OUTS["k_to_id"]
        n        = self.ARGS["n"]

        for i in range(n):
            if clts[i] is not None:
                clt = clts[i]
                pts = yx_edges[i].astype(np.int)
                # show = show_k[i]
                # which_id = (k_to_id * show)[clt]
                # which_id = k_to_id[clt]
                # pred[i][pts[:, 0], pts[:, 1]] = which_id
                pred[i][pts[:, 0], pts[:, 1]] = clts[i] + 1


    def save(self, h5):
        pickle.dump((self.ARGS, self.IMGS, self.OUTS), open(h5, "wb"))
