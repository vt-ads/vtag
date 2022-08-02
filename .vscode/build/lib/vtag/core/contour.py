# imports
import numpy  as np
import pandas as pd
import cv2 as cv
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

def bbox_img(img, bbox):
    x, y, w, h = bbox
    return img[y : y + h, x : x + w]

def get_mask(img, bbox, poi, win_smt=(5, 5), iter=5):
    """
    parameters
    ---
    img: 2d image array
    poi: n by 1 by 2 (xy)
    bbox: bounding box {x, y, w, h}
    win_smt: smoothing window
    iter: number of iteration for smoothing

    example
    ---
    vtag = VTag(k=4)
    vtag.load(path=path_dir)
    i = 0
    vtag.detect_poi(frame=i)
    poi  = vtag.poi(frame=i, k=0).reshape((-1, 1, 2))
    img  = vtag.img(i)
    bbox = cv.boundingRect(poi)
    img_mask = get_mask(img, bbox, poi, (5, 5), 5)
    plt.imshow(bbox_img(img, bbox) * img_mask)
    """
    img_mask = np.zeros(img.shape, dtype=np.uint8)
    # fill in contour areas
    cv.drawContours(img_mask, [poi], 0, (1), -1)
    kernel   = cv.getStructuringElement(cv.MORPH_RECT, win_smt)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel, iterations=iter)
    img_mask = (img_mask == 1)
    # return
    return bbox_img(img_mask, bbox)