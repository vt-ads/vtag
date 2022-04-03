from .utils import distance
import numpy  as np
import pandas as pd
import cv2    as cv
# surpress sklearn warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.cluster    import AgglomerativeClustering
from sklearn.neighbors  import kneighbors_graph

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
    centers: k by 2 (xy) array

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

def sort_points(pts1, pts2):
    """
    sort the points in pts1 to have minimum distance with points from pts2

    outputs
    ---
    pts1 : sorted pts1
    order: corresponding positions in pts2

    example
    ---
    pts1 = np.array([[10, 20], [3, 5], [9, 5], [4, 2]])
    pts2 = np.array([[3, 2], [12, 22], [3, 6], [8, 4]])
    pts1_sort, order = sort_points(pts1, pts2)
    # pts1_sort should be [[4, 2], [10, 20], [3, 5], [9, 5]]
    # order     should be [1, 2, 3, 0]

    # validation
    plt.plot(pts1[:, 0], pts1[:, 1])
    plt.plot(pts2[:, 0], pts2[:, 1])
    plt.plot(pts1_sort[:, 0], pts1_sort[:, 1])
    plt.plot(pts2[:, 0],      pts2[:, 1])
    """
    k     = len(pts1)
    order = np.arange(k) # k * (new position)
    for i in range(k):
        dist = [distance(pts1[j], pts2[i]) for j in range(i, k)]
        idx_src = np.argmin(dist) + i
        idx_dst = i
        # swap points
        val_tmp        = pts1[idx_dst].copy()
        pts1[idx_dst]  = pts1[idx_src]
        pts1[idx_src]  = val_tmp
        # swap order
        val_tmp        = order[idx_dst]
        order[idx_dst] = order[idx_src]
        order[idx_src] = val_tmp

    # return
    return pts1, np.argsort(order)
