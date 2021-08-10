from lib import *

class VTags():

    def __init__(self, k=1, n_tags=10):
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
                        pos_yx = None, # (y, x) coordinate of edges
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
            self.OUTS["pos_yx"] = n * [None]
            self.OUTS["k_to_id"]= np.zeros((n, self.ARGS["k"]))
            self.OUTS["pred"]   = np.zeros((n, h, w), dtype=np.uint8)
            self.OUTS["pred_cls"] = np.zeros((n, h, w), dtype=np.uint8)


    def run(self):
        self.detect_movements()
        self.detect_edges()
        self.detect_clusters()
        self.map_k_to_id()
        self.make_predictions()

    def detect_movements(self):
        '''
        '''
        imgs_mov = self.IMGS["mov"]
        imgs_bw  = self.IMGS["bw"]
        n        = self.ARGS["n"]

        # compute std images
        for i in range(n):
            imgs_mov[i] = detect_imgs(imgs_bw, i)

        # binarize
        std_matrix = imgs_mov[~np.isnan(imgs_mov).max(axis=(1, 2))]
        cutoff = np.median(std_matrix) + (3 * np.std(std_matrix))
        for i in range(n):
            imgs_mov[i] = get_binary(imgs_mov[i], cutabs=cutoff)

    def detect_edges(self, n_denoise=5):
        '''
        '''
        imgs_mov = self.IMGS["mov"]
        imgs_edg = self.IMGS["edg"]
        n        = self.ARGS["n"]

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

        for i in range(n):
            conv = convolve2d(imgs_mov[i], k_edge, mode="same")
            conv = get_binary(conv)
            for _ in range(n_denoise):
                conv = convolve2d(conv, k_gauss, mode="same")
                conv = get_binary(conv, cutabs=.5)
            imgs_edg[i] = conv

        # create reduncdent pixel
        imgs_edg[:, 10:20, 10:20] = 1 # in the feature it's (14, 14)

    def detect_clusters(self):
        '''
        '''
        imgs_edg = self.IMGS["edg"]
        clusters = self.OUTS["cls"]
        centers  = self.OUTS["cts"]
        pos_yx   = self.OUTS["pos_yx"]
        k        = self.ARGS["k"]
        n        = self.ARGS["n"]

        for i in range(n):
            try:
                cls, cts, np_yx = do_k_means(imgs_edg, i, k)
                clusters[i] = cls
                centers[i]  = cts
                pos_yx[i]   = np_yx
            except Exception as e:
                print(e)


    def map_k_to_id(self):
        '''
        '''
        n        = self.ARGS["n"]
        k        = self.ARGS["n_id"]
        features_all = self.OUTS["cts"]

        for i in range(n):
            self.OUTS["k_to_id"][i] = map_features_to_id(features_all[i], k)


    def make_predictions(self):
        '''
        '''
        clts     = self.OUTS["cls"]
        pos_yx   = self.OUTS["pos_yx"]
        k_to_id  = self.OUTS["k_to_id"]
        pred     = self.OUTS["pred"]
        pred_clt = self.OUTS["pred_cls"]

        n        = self.ARGS["n"]

        for i in range(n):
            if clts[i] is not None:
                clt = clts[i]
                pts = pos_yx[i].astype(np.int)

                # show prediction
                which_id = k_to_id[i][clt]
                pred[i][pts[:, 0], pts[:, 1]] = which_id

                ## show clusters
                pred_clt[i][pts[:, 0], pts[:, 1]] = clts[i] + 1


    def save(self, h5="model.h5"):
        pickle.dump((self.ARGS, self.IMGS, self.OUTS), open(h5, "wb"))
