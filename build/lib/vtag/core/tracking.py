import numpy     as np
import cv2       as cv
import pandas    as pd
from .utils import standardize_to_01

def track(tracker="SparseLK", **kwargs):
    """
    Calculates an optical flow for a sparse feature set using
    the iterative Lucas-Kanade method with pyramids.
    ---

    parameters
    ---
    tracker: tracker name

    return
    ---
    ls_pts: tracked points, n by k by 2 (xy)
    ls_err: errors, n by k
    """
    # tracker
    if tracker == "SparseLK":
        ls_pts, ls_err = tracker_SparseLK(**kwargs)
    else:
        ls_pts, ls_err = tracker_cv(tracker=tracker, **kwargs)

    # return
    return ls_pts, ls_err


def tracker_cv(tracker, imgs, pts_0, win_xy=(50, 50)):
    """
    imgs   : n x w x h numpy array
    pts_0  : starting tracking points, k by 2 (xy)
    """
    # get parameters
    n = len(imgs)
    k = len(pts_0)

    # create outputs
    ls_pts = np.zeros((n, k, 2))
    ls_err = np.zeros((n, k))
    ls_pts[0] = pts_0

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
        bbx_init = pt_to_bbx(pts_0[ik], win_xy)
        ok = tracker.init(imgs[0], bbx_init)

        for i in range(1, n):
            # track
            ok, bbx = tracker.update(imgs[i])
            # results
            ls_err[i, ik] = ok
            ls_pts[i, ik] = bbx_to_pt(bbx)

    ls_err = k - ls_err

    return ls_pts, ls_err

def tracker_SparseLK(imgs, pts_0, win_xy=(50, 50)):
    """
    imgs   : n x w x h numpy array
    pts_0  : starting tracking points, k by 2 (xy)
    """
    # get parameters
    n = len(imgs)
    k = len(pts_0)

    # create outputs
    ls_pts = np.zeros((n, k, 2))
    ls_err = np.zeros((n, k))
    ls_pts[0] = pts_0

    # reshape for LK
    pts_0 = pts_0.astype(np.float32).reshape((k, 1, 2))
    # swap pointers
    img_cur = imgs[0]
    pts_prv = pts_0
    pts_cur = pts_prv

    for i in range(1, n):
        # update pointers
        pts_prv = pts_cur
        img_prv = img_cur
        img_cur = imgs[i]
        # tracking
        pts_cur, _, err = cv.calcOpticalFlowPyrLK(img_prv, img_cur,
                                                  pts_prv, None,
                                                  winSize=win_xy)
        # results
        ls_err[i] = err.reshape(-1)
        ls_pts[i] = pts_cur.reshape(-1, 2)

    ls_err = standardize_to_01(ls_err)

    return ls_pts, ls_err

def pt_to_bbx(pt, win):
    """
    parameters
    ---
    pt  : centroids (x, y)
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

    return
    ---
    centroids of the bbx [x, y]
    """
    return [bbx[0] + (bbx[2] / 2),
            bbx[1] + (bbx[3] / 2)]

