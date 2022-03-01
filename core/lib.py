import pandas    as pd
import numpy     as np
import cv2       as cv

def get_binary(signals, cut=.5, cutabs=None, upbound=1):
    """
    parameters
    ---
    cut   : Use the quantile of input data as the threshold
    cutabs: Use absolute values as the threshold

    return
    ---
    signals with values of [0 or upbound] in the same dimensions
    """
    if cutabs is None:
        # set color to 1 if over 50 percentile
        _, out = cv.threshold(signals,
                        np.quantile(signals, cut), upbound, cv.THRESH_BINARY)
    else:
        # set color to 1 if over cutabs
        _, out = cv.threshold(signals,
                        cutabs, upbound, cv.THRESH_BINARY)
    return out

def load_labels(n_frames, k):
    try:
        labels = pd.read_csv("labels.csv")
        labels = lb_from_pd_to_np(labels)
    # see "lb_from_pd_to_np"
    except:
        labels = np.zeros((n_frames, k, 2), dtype=np.int)
    # if there is no "labels.csv" in current directory
    # create "labels" as n*k*2
    # n: number of frames/pictures
    # k: number of animals
    return labels
