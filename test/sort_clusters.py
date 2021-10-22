path_lib = "/Users/jchen/Dropbox/projects/Virtual_Tags/src/"
os.chdir(path_lib)
from lib import *
from VTags import VTags

# Input
dataname = "group_small"

# WD
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)

# resume
app = VTags(h5="model.h5")
app.create_labels()
app.save_labels()
app.save("model.h5")

# get images and clusters
pred = app.IMGS["pred"]
clts = make_labels(pred)

b = 61
e = 64
# for i in range(2):
#     plt.plot(clts[b:e, i, 1], clts[b:e, i, 0])
clusters = clts[b:e].copy()
n_frames, k, _ = clusters.shape
for i in range(n_frames):
    score_ori    = get_scores(clusters, i)
    clusters_alt = clusters.copy()
    for k1 in range(0, k - 1):
        for k2 in range(k1, k):
            clusters_alt[i] = swap_clusters(clusters[i], swp1=k1, swp2=k2)
            score_alt       = get_scores(clusters_alt, i)
            if score_alt > score_ori:
                clusters  = clusters_alt.copy()
                score_ori = score_alt
                # update images
                pred[pred == k1] = 9
                pred[pred == k2] = k1
                pred[pred == 9]  = k2


cltf = clts[b:e].copy()


# # complete run
# app = VTags(k=2, n_tags=20)
# bound_x = [180, 730, 725, 170]
# bound_y = [70,  90,  460, 440]
# bounds = np.array([[y, x] for x, y in zip(bound_x, bound_y)])
# app.load(bounds=bounds)
# app.detect_movements()
# app.detect_edges()
# app.detect_clusters()
# app.map_k_to_id()
# app.make_predictions()
# app.create_labels()
# app.save_labels()
# app.save("model.h5")
