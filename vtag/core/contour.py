# imports
import numpy  as np
import pandas as pd
from scipy.signal import convolve2d

# vtag functions
from .utils import get_binary

def detect_contour(imgs_poi, n_denoise=2):
    n = len(imgs_poi)
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

    imgs_poi_c = np.zeros(imgs_poi.shape)
    for i in range(n):
        # denoise the movement pictures
        conv = convolve2d(imgs_poi[i], k_gauss, mode="same")
        for _ in range(n_denoise):
            conv = convolve2d(conv, k_gauss, mode="same")
        conv = get_binary(conv, cutabs=.5)
        # perform edge detection
        conv = convolve2d(conv, k_edge, mode="same")
        conv = get_binary(conv, cutabs=.5)
        imgs_poi_c[i] = conv

    return imgs_poi_c
