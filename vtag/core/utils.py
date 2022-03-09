"""
Functions that are used by multiple files are placed in lib.py
"""

# imports
import numpy     as np
import cv2       as cv
import pandas    as pd
import matplotlib.pyplot as plt
from scipy.signal import spline_filter, convolve, convolve2d, find_peaks


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
    signals = np.array(signals, dtype=np.float32)
    if cutabs is None:
        # set color to 1 if over 50 percentile
        _, out = cv.threshold(signals,
                        np.quantile(signals, cut), upbound, cv.THRESH_BINARY)
    else:
        # set color to 1 if over cutabs
        _, out = cv.threshold(signals,
                        cutabs, upbound, cv.THRESH_BINARY)
    return out

def get_nonzero_from_img(img):
    """
    Turn a sparse 2d array to a dataframe storing the positions of non-zero pixels
    """
    f, y, x = np.nonzero(img)
    return pd.DataFrame({"frame": f, "y": y, "x" : x})

def show_poi(img, poi, lbs=None, figsize=None):
    """
    inputs
    ---
    img: 2d np array
    poi: dataframe (x, y)
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(img, aspect='auto')
    if lbs is None:
        plt.scatter(poi["x"], poi["y"], s=3)
    else:
        for lb in np.unique(lbs):
            plt.scatter(poi[lbs == lb]["x"],
                        poi[lbs == lb]["y"], 3)

