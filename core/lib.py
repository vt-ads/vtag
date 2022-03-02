import pandas    as pd
import numpy     as np
import cv2       as cv



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
