import numpy as np
import cv2   as cv
from .utils import get_binary

def detect_motion(imgs, frame, span=1):
    """
    parameters
    ---
    imgs : black and white pictures, n*h*w
    frame: index of frame/picture

    return
    ___
    motion itensity in the same dimensions
    """
    i = frame
    j = span
    # stack 2*2 iamges: out_img is a 4-frame std image
    out_std = []
    # iterate twice, 2 things in each iteration
    for _ in range(2):
        out_std += [
            imgs[i:  (i+j+1)].std(axis=(0, )),
            imgs[(i-j):(i+1)].std(axis=(0, ))
        ]
        j += 1

    # average over the 4 things
    out_img = sum(out_std) / len(out_std)
    return out_img

def get_threshold_motion(imgs_motion, n_ticks=3):
    """
    calculate threshold to filter high-motion pixel

    return
    ---
    cutoff : value to fileter POI
    tick   : s.d. of motion distribution
    """
    nonna_frames = imgs_motion[~np.isnan(imgs_motion).max(axis=(1, 2))]
    tick   = np.std(nonna_frames)
    # median + n_ticks=3 SD (99.7%)
    cutoff = np.median(nonna_frames) + (n_ticks * tick)
    return cutoff, tick

def rescue_low_motion_frame(imgs_poi, imgs_motion, cutoff, tick, rate_rescue=.3):
    """
    inputs
    ---
    cutoff, tick: computed from `get_threshold_motion()`

    return
    ---
    NULL, update imgs_poi
    """
    # count the number of nonzero elements in each frame [n-length vector]
    nsig_frames = np.array([np.count_nonzero(img) for img in imgs_poi])
    # set up threshold for rescuing frame with motion below rate_rescue (0.3) quantile
    cut_rescue  = np.quantile(nsig_frames, rate_rescue)
    # extract the index of frames where nonzero elements are below the threshold
    idx_rescue  = np.where((nsig_frames < cut_rescue) & (nsig_frames > 0))[0]
    for i in idx_rescue:
        adjust  = 0
        # keep looping if the number of nonzero elements is still below threshold
        while np.count_nonzero(imgs_poi[i]) <= cut_rescue:
            adjust += (tick * 0.2)
            # decrease the threshold a little bit when changing into black and white
            imgs_poi[i] = get_binary(imgs_motion[i], cutabs=cutoff - adjust)

def add_vision_persistence(imgs_poi):
    # add the nonzero pixels (value=1) from the next frame to the current one
    imgs_poi[:-1] += (imgs_poi[1:] == 1)
    # add the nonzero pixels (value=1) from the previous frame to the current one
    imgs_poi[1:]  += (imgs_poi[:-1] == 1)

    max_value = 1 * 2 + 1
    cut       = max_value * 0.3 # [1, 0, 0] -> [1, 1, 0] one side
    # change pictures into binary again
    for i in range(len(imgs_poi)):
        imgs_poi[i] = get_binary(imgs_poi[i], cutabs=cut)

