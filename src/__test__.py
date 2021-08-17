from pyqtgraph.Qt import scale
from lib import *
from Tags import VTags

# Input
# dataname = "one_pig"
dataname = "group"

# WD
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)

# Detailed run
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
# app.save("model_24f_20t.h5")

# # First Run
# app = VTags(k=1, n_tags=20)
# app.load()
# app.run()
# app.save("model2.h5")

## Detail run small
# app = VTags(k=1, n_tags=10)
# app.load()
# app.detect_movements()
# app.detect_edges()
# app.save("model.h5")

# Resume run
app = VTags(k=2)
app.load(h5="model_24f_20t.h5")


imgs = app.IMGS["bw"][40]

plt.imshow(imgs)

### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#NOTE: arbitrary numbers
# movement cut: mean + 3std
# movement res: 30%

# frame: 228 - 230
# zero-size array to reduction operation maximum which has no identity
# frame: 257 - 262, 270 - 276,
# could not broadcast input array from shape (28) into shape (32)
# frame: 297 - 299
# could not broadcast input array from shape (24) into shape (32)

# problem frames
# 15, 206

### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

np.array(dt)
def load_labels(n_frames):
    try:
        labels = pd.read_csv("labels.csv")
        labels = np.array(labels)
    except:
        labels = np.zeros((n_frames, 2), dtype=np.int)

    return labels

labels = np.zeros((10, 6))

x = []
y = []

idx_x = [2 * i for i in range(3)]
idx_y = [2 * i + 1 for i in range(3)]
x = labels[3, idx_x]
x = labels[3, idx_y]



    labels[3] = 3, 6
    dt = pd.DataFrame(labels)
    dt.columns = ['x', 'y']
    dt.to_csv("labels.csv", index=False)




# Crop
app.map_k_to_id()
app.make_predictions()
app.save("model_24f_20t.h5")


i = 187
n        = app.ARGS["n"]
k        = app.ARGS["k"]
imgs_edg = app.IMGS["edg"]
cls      = app.OUTS["cls"]
cts      = app.OUTS["cts"]
pos_yx   = app.OUTS["pos_yx"]
kid   = app.OUTS["k_to_id"]
pcs_all  = app.OUTS["pcs"]
img_e = app.IMGS["edg"]
img_p = app.OUTS["pred"]
img_c = app.OUTS["pred_cls"]


k = app.ARGS["n_id"]
features = cts[i]
pcs = pcs_all[i]


plt.scatter(pcs[:, 0], pcs[:,  1])


n_ft = len(features)
new_ids = np.array([0] * n_ft)


#-- Get PCs from features, and cluster into k+1 groups
pca = PCA(k)
pca.fit(features)
pcs = pca.transform(features) * pca.explained_variance_ratio_


ids, _ = cv_k_means(features, k)
# ids = cluster_gm(features, k)


gm = GaussianMixture(n_components=k,
                     max_iter=100,
                     n_init=10,
                    #  weights_init=weights,
                     init_params="kmeans",
                     tol=1e-4)
ids =  gm.fit_predict(pcs)


plt.scatter(pcs[ids == 0, 0], pcs[ids == 0, 1])
plt.scatter(pcs[ids == 1, 0], pcs[ids == 1, 1])




# get collection of cluster numbers
value_counts = pd.value_counts(ids)
keys = value_counts.keys()

#-- clean outliers and include missed
major = keys[np.where(value_counts == max(value_counts))[0][0]]

# remove outliers
idx_maj = np.where(ids == major)[0]
pts_maj, keep_idx_maj = remove_outliers(pcs[idx_maj])

# update majority idx
idx_out = idx_maj[~keep_idx_maj]
idx_maj = idx_maj[keep_idx_maj]

# new center of majority
mid_maj = np.median(pts_maj, axis=0)

# distance to the center of each points
dist = np.array([distance(pcs[i], mid_maj) for i in range(n_ft)])

# either belong to major group (1) or not (0)
ids_tmp, _ = cv_k_means(dist, 2)
ids_tmp = reassign_id_by(ids_tmp, dist, by="value")

new_ids[ids_tmp == 1] = major
new_ids[idx_out] = -1

# finalize new ids
new_ids = reassign_id_by(new_ids, values=pcs, by="size")
new_ids[idx_out] = 0


new_ids

plt.scatter(pcs[ids == 0, 0], pcs[ids == 0, 1])
plt.scatter(pcs[ids == 1, 0], pcs[ids == 1, 1])
plt.scatter(pcs[ids == 2, 0], pcs[ids == 2, 1])
