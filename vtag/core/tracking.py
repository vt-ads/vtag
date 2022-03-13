import numpy     as np
import cv2       as cv
import pandas    as pd
from sklearn.cluster    import AgglomerativeClustering
from sklearn.neighbors  import kneighbors_graph

def track(st, ed, tracker="CSRT", **kwargs):
    """
    Calculates an optical flow for a sparse feature set using
    the iterative Lucas-Kanade method with pyramids.
    ---

    parameters
    ---
    imgs   : n x w x h numpy array
    idx    : idx th frame to generate pts_idx
    pts_idx: starting tracking points, k by 2 (xy)
    st, ed : start and ending frames for tracking

    return
    ---
    tracked points: n by k by 2 (xy)
    error list    : n by k
    """
    # define iterator (use list() for multiple uses)
    is_reversed = st > ed
    if is_reversed:
        iterator = list(enumerate(reversed(range(ed, st))))
    else:
        iterator = list(enumerate(range(st, ed)))

    # tracker
    if tracker == "SparseLK":
        ls_pts, ls_err = tracker_SparseLK(iterator=iterator, **kwargs)
    else:
        ls_pts, ls_err = tracker_cv(tracker=tracker, iterator=iterator, **kwargs)

    # return
    if is_reversed:
        return np.flip(ls_pts, axis=0), np.flip(ls_err, axis=0)
    else:
        return ls_pts, ls_err


def tracker_cv(tracker, imgs, idx, pts_idx, iterator, winSize=(50, 50)):
    """
    """
    # get parameters
    n = len(iterator)
    k = len(pts_idx)

    # create outputs
    ls_pts = np.zeros((n, k, 2))
    ls_err = np.zeros((n, k))

    for ik in range(k):
        # select tracker
        if tracker == "CSRT":
            tracker = cv.TrackerCSRT_create()
        elif tracker == "MedianFLow":
            tracker = cv.legacy.TrackerMedianFlow_create()
        elif tracker == "mosse":
            tracker = cv.legacy.TrackerMOSSE_create()
        elif tracker == "MIL":
            tracker = cv.TrackerMIL_create()

        # initialize tracker
        bbx_init = pt_to_bbx(pts_idx[ik], winSize)
        ok = tracker.init(imgs[idx], bbx_init)

        for idx_out, i in iterator:
            # track
            ok, bbx = tracker.update(imgs[i])
            # results
            ls_err[idx_out, ik] = ok
            ls_pts[idx_out, ik] = bbx_to_pt(bbx)

    return ls_pts, ls_err

def tracker_SparseLK(imgs, idx, pts_idx, iterator, winSize=(50, 50)):
    """
    """
    # get parameters
    n = len(iterator)
    k = len(pts_idx)

    # create outputs
    ls_pts = np.zeros((n, k, 2))
    ls_err = np.zeros((n, k))

    # reshape for LK
    pts_idx = pts_idx.astype(np.float32).reshape((k, 1, 2))
    # swap pointers
    img_cur = imgs[idx]
    pts_prv = pts_idx
    pts_cur = pts_prv

    for idx_out, i in iterator:
        # update pointers
        pts_prv = pts_cur
        img_prv = img_cur
        img_cur = imgs[i]
        # tracking
        pts_cur, _, err = cv.calcOpticalFlowPyrLK(img_prv, img_cur,
                                                  pts_prv, None,
                                                  winSize=winSize)
        # results
        ls_err[idx_out] = err.reshape(-1)
        ls_pts[idx_out] = pts_cur.reshape(-1, 2)

    return ls_pts, ls_err

def pt_to_bbx(pt, win):
    """
    parameters
    ---
    pt  : (x, y)
    win : (size_w, size_h), window size

    return
    ---
    bbx : (x, y, w, h)
    """
    return (int(pt[0] - (win[0] / 2)), # x
            int(pt[1] - (win[1] / 2)), # y
            win[0], win[1]) # w, h

def bbx_to_pt(bbx):
    """
    parameter
    ---
    bbx : (x, y, w, h)
    """
    return [bbx[0] + (bbx[2] / 2),
            bbx[1] + (bbx[3] / 2)]

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

