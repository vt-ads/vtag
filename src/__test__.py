from pyqtgraph.Qt import scale
from lib import *
from Tags import VTags

# Input
# dataname = "one_pig"
dataname = "group_small"

# WD
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)

# Detailed run
# app = VTags(k=2, n_tags=10)
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

# # First Run
# app = VTags(k=1, n_tags=20)
# app.load()
# app.run()
# app.save("model.h5")

## Detail run small
# app = VTags(k=1, n_tags=10)
# app.load()
# app.detect_movements()
# app.detect_edges()

# Resume run
# app = VTags(k=2)
# app.load(h5="model.h5")

### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# generate features
features = app.OUTS["features"]


os.chdir(path_project + "group")
pre_grp = np.array(pd.read_csv("labels.csv")).reshape((30, 2, 2))
dist = np.array([distance(p1, p2) for p1, p2 in pre_grp])
len(dist)


### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# ground truth
lbs = np.array([pd.read_csv("truth/labels_1.csv"),
                pd.read_csv("truth/labels_2.csv"),
                pd.read_csv("truth/labels_3.csv")])

plt.plot(lbs[:, :, 0].std(axis=0))
plt.plot(lbs[:, :, 1].std(axis=0))
plt.plot(lbs[:, :, 2].std(axis=0))
plt.plot(lbs[:, :, 3].std(axis=0))


np.where(lbs[:, :, 0].std(axis=0) > 25)
### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

features = app.OUTS["features"]
i = 157
fts = features[157]
fts.shape

plt.imshow(fts.transpose())



app.make_predictions()
app.create_labels()
app.save("model_24f_20t.h5")


imgs = app.IMGS["pred"]
img = imgs[295]

plt.imshow(img[420:430, 300:310])
img_sub = img[420:430, 300:310].copy()

img_sub[img_sub == 1] = 9


np.sum(img==-1)



ids = app.OUTS["k_to_id"]
k   = app.ARGS["n_id"]
ki = 1

app.save()

d

x = [[1, 2, 3, 4], [5, 6, 7, 8]]
lbs = lb_from_pd_to_np(x)

lbs[1, :, 0]

app.OUTS["pred_labels"] = make_labels(imgs_p)

'''
NOTE: can be sort by colors and coordinate
Sort k centers in cts[i + 1] based on distance between (i) and (i + 1)
args
    i  : ith frame
Time complexity O = (k-1)^2
'''



imgs = app.IMGS["pred"]
clusters = make_labels(imgs_p)
app.OUTS["pred_labels"] = sort_clusters(clusters, imgs)
app.save("model_24f_20t.h5")

x = [3, 6]
y = [1, 2]
tmp = np.array([x, y]).swapaxes(0, 1)[:, [1, 0]]

lb_from_np_to_pd(tmp)

tmp.shape

imgs = app.IMGS["bw"][40]

plt.imshow(imgs)



value_min = idx_min + 1 # 0 -> 1, 1 -> 2
value_k1 = k1 + 1
imgs[imgs == value_min] = -1
imgs[imgs == value_k1] = value_min
imgs[imgs == -1] = value_k1



img = imgs[i + 1]
[imgs == ]


# swap clustering i+1
if self.OUTS["cls"][i + 1] is not None:
    pos_k1 = self.OUTS["cls"][i + 1] == k1
    pos_min = self.OUTS["cls"][i + 1] == idx_min
    self.OUTS["cls"][i + 1][pos_k1] = idx_min
    self.OUTS["cls"][i + 1][pos_min] = k1

# compute the distance of the last k
ls_min += [distance(self.OUTS["features"][i][k - 1],
                self.OUTS["features"][i + 1][k - 1])]
# compute distances
self.OUTS["dist"][i + 1] = ls_min
self.OUTS["dct"][i + 1]  = self.OUTS["features"][i + 1] - self.OUTS["features"][i]

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
cts      = app.OUTS["features"]
pos_yx   = app.OUTS["pos_yx"]
kid   = app.OUTS["k_to_id"]
pcs_all  = app.OUTS["pcs"]
img_e = app.IMGS["edg"]
img_p = app.IMGS["pred"]
img_c = app.IMGS["pred_cls"]


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
