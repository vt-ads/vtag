import matplotlib.pyplot as plt
import numpy as np

def show_poi(img, poi, figsize=None):
    """
    inputs
    ---
    img: 2d np array
    poi: dataframe (x, y)

    example
    ---
    i = 0
    img = vtag.img(i)
    poi = vtag.poik(i)
    show_poi(img, poi)
    """
    x = poi[:, 0]
    y = poi[:, 1]
    k = poi[:, 2]
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(img, aspect='auto')
    for lb in np.unique(k):
        plt.scatter(x[k == lb],
                    y[k == lb], s=3)
