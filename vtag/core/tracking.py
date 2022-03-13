import numpy     as np
import cv2       as cv
import pandas    as pd
from sklearn.cluster    import AgglomerativeClustering
from sklearn.neighbors  import kneighbors_graph

def LK_tracking(imgs, idx, pts_idx, st, ed):
    """
    Calculates an optical flow for a sparse feature set using
    the iterative Lucas-Kanade method with pyramids.
    ---

    parameters
    ---
    imgs   : n x w x h numpy array
    idx    : idx th frame to generate pts_idx
    pts_idx: starting tracking points, k by 1 by 2 (xy)
    st, ed : start and ending indexes for tracking

    return
    ---
    tracked points: n by k by 2 (xy)
    error list    : n by k
    """
    # reshape points
    n       = abs(ed - st)
    k       = pts_idx.shape[0]
    pts_idx = pts_idx.astype(np.float32).reshape((k, 1, 2))

    # create outputs
    ls_pts = np.zeros((n, k, 2))
    ls_err = np.zeros((n, k))

    # swap pointers
    img_cur = imgs[idx]
    pts_prv = pts_idx
    pts_cur = pts_prv

    # tracking
    is_reversed = st > ed
    if is_reversed:
        iterator = enumerate(reversed(range(ed, st)))
    else:
        iterator = enumerate(range(st, ed))
    for idx_out, i in iterator:
        # update pointers
        pts_prv = pts_cur
        img_prv = img_cur
        img_cur = imgs[i]
        # tracking
        pts_cur, _, err = cv.calcOpticalFlowPyrLK(img_prv, img_cur,
                                                  pts_prv, None,
                                                  winSize=(50, 50))
        # results
        ls_err[idx_out] = err.reshape(-1)
        ls_pts[idx_out] = pts_cur.reshape(-1, 2)

    # return
    if is_reversed:
        return np.flip(ls_pts, axis=0), np.flip(ls_err, axis=0)
    else:
        return ls_pts, ls_err

def cluster_poi(data, k, method="cv"):
    """
    imputs
    ---
    data : n by 2(x, y) 2d-array, or pandas
    k    : number of clusters

    parameters
    ---
    method: "cv", run OpenCV k-means; "agglo", run sklearn.cluster

    outputs
    ---
    lbs: a vector of size n showing clustering labels.

    """
    data = np.array(data).astype(np.float32)

    if method == "cv":
        criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        param_k = dict(data=data,
                    K=k,
                    bestLabels=None,
                    criteria=criteria,
                    attempts=10,
                    flags=cv.KMEANS_PP_CENTERS)
        _, lbs, centers = cv.kmeans(**param_k)
        lbs = lbs.reshape(-1)
        # return
        return lbs, centers

    elif method == "agglo":
        knn_graph = kneighbors_graph(data, 30, include_self=False)
        model     = AgglomerativeClustering(linkage="ward",  # average, complete, ward, single
                                            connectivity=knn_graph,
                                            n_clusters=k)
        model.fit(data)
        # get output labels
        lbs = model.labels_
        # computer centroids
        data = pd.DataFrame(data)
        data.loc[:, "k"] = lbs
        centers = np.array(data.groupby(["k"]).aggregate("median"))
        # return
        return lbs, centers

