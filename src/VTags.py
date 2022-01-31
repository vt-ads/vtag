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
        # number of rows, columns and channels
        # they are the same for all pictures 
        # c is usually 3 for colorful pictures 

        # create np matrix
        if n == -1:
            n = len(files)
        # n is the number of frames/pictures
        imgs_rgb = np.zeros((n, h, w, c), dtype=np.uint8)
        # create a 4-dimensional ndarray for the video recording 

        # iterate through files
        for i in range(n):
            imgs_rgb[i] = cv.imread(files[i])
        # fill in the 4-dimensional ndarray with their values

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
        # n*h*w

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
            # denoise the movement pictures

            conv = get_binary(conv, cutabs=.5)
            conv = convolve2d(conv, k_edge, mode="same")
            # perform edge detection 

            conv = get_binary(conv, cutabs=.5)
            imgs_edg[i] = conv

            # create fake signals to avoid empty array error
            imgs_edg[i, 5:10, 5:10] = 1
            # find pos of edges and filter edges by safe area (boundary)
            pos_yx_tmp = find_nonzeros(imgs_edg[i])
            # get locations and/or numeric color of non-zero pixels
            # number of non-zero pixels * 3
            pos_yx[i]  = filter_edges(pos_yx_tmp, bounds)
            # the location of non-zero pixels within the bounds


    def detect_clusters(self):
        '''
        '''
        n        = self.ARGS["n"]
        # number of frames/picture
        k        = self.ARGS["k"]
        # number of clusters
        imgs_edg = self.IMGS["edg"]
        # detected edges, n*h*w
        clusters = self.OUTS["cls"]
        # n * [None]
        pos_yx   = self.OUTS["pos_yx"]
        # n * number of non-zero pixels * 3
        # 24 is: time(8)+spatial(4)+xy(12))
        centers  = np.zeros((n, k, 24))

        for i in range(n):
        # for every frame
            try:
                cls, cts, np_yx = do_k_means(imgs_edg, pos_yx[i], i, k)
                clusters[i] = cls
                centers[i]  = cts
                pos_yx[i]   = np_yx
                # clusters: number of interesting pixels * 1 (label of cluster)
                # centers: k*24, the average across the features for the interesting pixels and coordinates for each cluster
                # yx: number of interesting pixels * 2 (coordinates of interesting pixels)
            except Exception as e:
                print(e)

        self.OUTS["features"] = centers

    def map_k_to_id(self):
        '''
        '''
        n            = self.ARGS["n"]
        # number of frames/pictures
        k            = self.ARGS["n_id"]
        # number of animals 
        features_all = self.OUTS["features"]
        # the average across the features for the interesting pixels and coordinates for each cluster

        for i in range(n):
        # for every frame
            self.OUTS["k_to_id"][i], self.OUTS["pcs"][i] =\
                map_features_to_id(features_all[i], k)
                # features_all: n*k*24
        # k_to_id: n * k
        # pcs: n * k * 2 

    def make_predictions(self):
        '''
        '''
        # number of frames
        n        = self.ARGS["n"]
        # number of animals
        k        = self.ARGS["n_id"]
        clts     = self.OUTS["cls"]
        # n*number of interesting pixels, the cluster label of each pixel
        pos_yx   = self.OUTS["pos_yx"]
        # n*number of interesting pixels*2, the coordiantes of pixels
        k_to_id  = self.OUTS["k_to_id"]
        # n*k, the animal label of each cluster
        pred     = self.IMGS["pred"]
        # n*h*w zeros 
        pred_clt = self.IMGS["pred_cls"]
        # n*h*w zeros 

        for i in range(n):
            if clts[i] is not None:
                clt = clts[i]
                # the cluster label each pixel is assigned to 
                pts = pos_yx[i].astype(np.int)

                # show prediction
                which_id = k_to_id[i][clt]
                # this maps every interesting pixel to animal
                # has length of interesting pixels 
                pred[i][pts[:, 0], pts[:, 1]] = which_id
                # for the interesting pixels in the image (pred), fill in the pig id

                # show clusters
                pred_clt[i][pts[:, 0], pts[:, 1]] = clt + 1 # cluster from 1 to k
                # for the interesting pixels in the image (pred_clt), 
                # fill in the cluster id   

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
        # n*h*w 
        clusters = make_labels(pred)
        # n*number of animals*2 
        # the median coordinates of pixels of the same animal 
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


