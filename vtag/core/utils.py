"""
Functions that are used by multiple files are placed in lib.py
"""

# imports
import numpy     as np
import cv2       as cv
import pandas    as pd
import matplotlib.pyplot as plt
import os


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

def standardize_to_01(matrix, axis=None):
    if axis is None:
        return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    else:
        return (matrix - np.min(matrix, axis=axis)) / (np.max(matrix, axis=axis) - np.min(matrix, axis=axis))

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** .5
