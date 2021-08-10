from lib import *
from Tags import VTags

# Input
dataname  = "one_pig"

# WD
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)

## Detailed run
# self.detect_movements()
# self.detect_edges()
# self.detect_clusters()
# self.map_k_to_id()
# self.make_predictions()

# First Run
# app = VTags(k=1, n_tags=10)
# app.load()
# app.run()
# app.save("model.h5")

# Detail run small
# app = VTags(k=1, n_tags=10)
# app.load()
# app.detect_movements()
# app.detect_edges()
# app.save("model.h5")

# Resume run
app = VTags(k=1)
app.load(h5="model.h5")

# Test features
app.map_k_to_id()
app.make_predictions()
app.save()


number_group = QtGui.QButtonGroup(QWidget())  # Number group
r0=QtGui.QRadioButton("0")
number_group.addButton(r0)
r1=QtGui.QRadioButton("1")
number_group.addButton(r1)
layout.addWidget(r0)

toggles
dict(edges=QRadioButton("Edges"),
     cls=QRadioButton("Clusters"),
     pre=QRadioButton("Predictions"))




i, k = 283, 1

features = app.OUTS["cts"]

i = 12
plt.imshow(app.OUTS["pred_cls"][i])

for i in range(300):
    map_features_to_id(features[i], k)


pca = PCA(2)
pca.fit(features[i])
pcs = pca.transform(features[i]) * pca.explained_variance_ratio_

ids, _ = cv_k_means(pcs, k + 1)

# get collection of cluster numbers
value_counts = pd.value_counts(ids)
keys = value_counts.keys()

n_ft = len(features[i])
new_ids = np.array([0] * n_ft)

major = 0
#-- clean outliers and include missed
for major in keys:
    # remove outliers
    idx_maj = np.where(ids == major)[0]

    if len(idx_maj) == 1:
        new_ids[idx_maj] = major

    else:
        pts_maj, keep_idx_maj = remove_outliers(pcs[idx_maj])

        # update majority idx
        idx_out = idx_maj[~keep_idx_maj]
        idx_maj = idx_maj[keep_idx_maj]

        # new center of majority
        mid_maj = np.median(pts_maj, axis=0)

        # distance to the center of each points
        dist    = np.array([distance(pcs[i], mid_maj) for i in range(len(pcs))])

        ids_tmp, _ = cv_k_means(dist, 2) # either belong to major group (1) or not (0)
        ids_tmp    = reassign_id_by(ids_tmp, dist, by="value")

        new_ids[ids_tmp == 1] = major
        new_ids[idx_out] = -1


new_ids = reassign_id_by(new_ids, values=pcs, by="size")
new_ids[idx_out] = 0



plt.scatter(pcs[ids==0, 0], pcs[ids==0, 1])
plt.scatter(pcs[ids==1, 0], pcs[ids==1, 1])

plt.scatter(pcs[new_ids==0, 0], pcs[new_ids==0, 1])
plt.scatter(pcs[new_ids==1, 0], pcs[new_ids==1, 1])





pcs = PCA(2).fit_transform(features[i])
ids, _ = cv_k_means(pcs, k + 1)




major = 0
# remove outliers
idx_maj = np.where(ids==major)[0]
pts_maj, keep_maj = remove_outliers(pcs[idx_maj])

# update idx
idx_maj = idx_maj[keep_maj]

mid_maj = np.median(pts_maj, axis=0)
dist = np.array([distance(pcs[i], mid_maj) for i in range(len(pcs))])

dist[idx_maj]
plt.scatter(range(10), dist)

new_ids, _ = cv_k_means(dist, 2)



n_k = pd.value_counts(new_ids).keys()
dist = []
for i in n_k:
    pts = pcs[new_ids == i]
    pt_ct = np.median(pts, axis=0)
    dist_pts = []
    for pt in pts:
        dist_pts += [distance(pt, pt_ct)]
    dist += [np.mean(dist_pts)]







plt.scatter(pcs[new_ids==0, 0], pcs[new_ids==0, 1])
plt.scatter(pcs[new_ids==1, 0], pcs[new_ids==1, 1])



# app.map_k_to_id()


#####
plt.imshow(app.IMGS["edg"][30])

plt.imshow(app.IMGS["edg"][30][120:130, 340:350])

app.IMGS["edg"][30][120:130, 340:350]
img = app.IMGS["edg"][30]
find_nonzeros(img)


app.OUTS["cls"]
xy = app.OUTS["pos_yx"]
np.sum(xy[4] == 0)




n= app.ARGS["n"]
k = app.ARGS["n_id"]
features_all = app.OUTS["cts"]
edd = app.IMGS["edg"]
edd[:, 0, 0] = 1


edd[30]
i = 30

clusters = map_features_to_id(features_all[i], k)
c1 = features_all[i][clusters==0].mean(axis=0)
c2 = features_all[i][clusters==1].mean(axis=0)


idx_sel = [12, 13]
dist = []
for i in range(300):
    clusters = map_features_to_id(features_all[i], k, use_pca=False)
    c1 = features_all[i][clusters == 0][:, idx_sel].mean(axis=0)
    c2 = features_all[i][clusters == 1][:, idx_sel].mean(axis=0)
    dist += [distance(c1, c2)]


plt.plot(dist)
dist = np.array(dist)
np.median(dist[dist!=0])
np.std(dist[dist!=0])
dist[131]


# re-order, put minority into backgorund(0)
# count values
value_counts = pd.value_counts(ids)

# find which key occur minimum
idx_min = np.where(value_counts == min(value_counts))[0][0]
keys = value_counts.keys()
key_min = keys[idx_min]
key_rest = keys[keys != key_min]

# re-assign
new_ids = np.array([0] * len(ids))
for i in range(len(key_rest)):
    assign_key = key_rest[i]
    assign_pos = np.where(ids == assign_key)[0]
    new_ids[assign_pos] = i + 1


    



clts     = app.OUTS["cls"]
pos_yx = app.OUTS["pos_yx"]
k_to_id  = app.OUTS["k_to_id"]
pred     = app.OUTS["pred"]
n = app.ARGS["n"]
i = 30

clt = clts[i]
pts = pos_yx[i].astype(np.int)
# show = show_k[i]
# which_id = (k_to_id * show)[clt]
which_id = k_to_id[i][clt]
pred[i][pts[:, 0], pts[:, 1]] = which_id

which_id.shape


plt.imshow(app.IMGS[""])


imgs = app.IMGS["edg"]


do_k_means(imgs, 50, 10)

features = app.OUTS["cts"]

map_features_to_id(features[0], 1)


i = 0
k = 10










# for i in range(len(imgs)):
for i in range(46, 49):
    pos_yx = find_nonzeros(imgs[i])
    n = len(pos_yx)
    feature_length = 8 + 4 + 2 # tp(8) + sp(4) + pos(2)
    dt_features = np.zeros((n, feature_length), dtype=np.int)

    for j in range(n):
        pos_y, pos_x = pos_yx[j]
        block_sp = make_block(imgs, i, (pos_y, pos_x), size=(3, 3))
        block_tp = make_block(imgs, i, (pos_y, pos_x), size=(2, 2, 2))
        if len(block_sp)
        ft_tp    = extract_features(block_tp, conv_type="temporal")
        ft_sp    = extract_features(block_sp, conv_type="spatial")
        dt_features[j] = np.concatenate([ft_tp, ft_sp, pos_yx[j]])

    dt_features = dt_features.astype(np.float32)


pos_y, pos_x = (0, 439)
bin_t, bin_y, bin_x = (3, 3, 3)
imgs[(i - bin_t): (i + bin_t + 1),
     (pos_y - bin_y): (pos_y + bin_y + 1),
     (pos_x - bin_x): (pos_x + bin_x + 1)]





i=47
make_block(imgs, i, (0, 439), size=(3, 3))
pos_y, pos_x = pos

 bin_t, bin_y, bin_x = size
        block = inputs[(i - bin_t): (i + bin_t + 1),
                       (pos_y - bin_y): (pos_y + bin_y + 1),
                       (pos_x - bin_x): (pos_x + bin_x + 1)]






clusters, centers = cv_k_means(dt_features, 5)
clusters






# for presentation
list(app.IMGS.keys())

i = 37

plt.imshow(app.IMGS["rgb"][i-2])
plt.imshow(app.IMGS["rgb"][i-1])
plt.imshow(app.IMGS["rgb"][i])
plt.imshow(app.IMGS["rgb"][i+1])
plt.imshow(app.IMGS["rgb"][i+2])

plt.imshow(np.std(app.IMGS["bw"][(i-2):(i+1)], axis=0))
plt.imshow(np.std(app.IMGS["bw"][(i-1):(i+1)], axis=0))
plt.imshow(np.std(app.IMGS["bw"][(i):(i+2)], axis=0))
plt.imshow(np.std(app.IMGS["bw"][(i):(i+3)], axis=0))



plt.imshow(app.IMGS["mov"][i])
plt.imshow(app.IMGS["edg"][i])




fts_dir = np.swapaxes(app.OUTS["dct"], 1, 0).\
    reshape((app.ARGS["k"], -1)).\
    astype(np.float32)
fts_yx = np.swapaxes(app.OUTS["cts"], 1, 0).\
    reshape((app.ARGS["k"], -1)).\
    astype(np.float32)
fts_dis = np.swapaxes(app.OUTS["dist"], 1, 0).\
    reshape((app.ARGS["k"], -1)).\
    astype(np.float32)

# fts = np.concatenate((fts_dir, fts_yx, fts_dis), axis=1)
fts = np.concatenate((fts_dir, fts_dis), axis=1)
ftsst = (fts - np.mean(fts, axis=0)) / (np.std(fts, axis=0) + 1e-9)


k_to_id, _ = cv_k_means(ftsst, app.ARGS["n_id"] + 1)
k_to_id += 1  # which k cluster -> which id
app.OUTS["k_to_id"] = k_to_id
app.make_predictions()

i = 35
plt.imshow(app.OUTS["cls"][i]==1)
plt.imshow(app.OUTS["pred"][i]==2)
plt.imshow(app.OUTS["pred"][i]==3)


app.OUTS["cls"]



# 63, 136

paths  = ls_files(path_project)
imgs   = load_np(paths)
imgsc  = load_np(paths, is_BW=False)
n      = len(imgs)


kernel = [[-1, -1, -1],
          [-1, 8, -1],
          [-1, -1, -1]]
gauss = np.array((
    [1, 4, 1],
    [4, 9, 4],
    [1, 4, 1]), dtype='int') / 29


# parameters
n_id   = 2
n_tags = 10
k      = n_id * n_tags

# Outputs
OUT = dict()
OUT["n"] = n
OUT["k"] = k
# images 2D array
OUT["shades"]   = []
OUT["edges"]    = []
# n-length list
OUT["centers"]  = np.zeros((n, k, 2)) # (y, x)
OUT["clusters"] = n * [None]
OUT["yx_edges"] = n * [None]

for i in range(n):
# for i in range(10):
    if i % 10 == 0:
        print("%d\r" % (i))

    # === === === === === Find Candidates === === === === ===
    # Detect candidate pixels
    pred = detect_imgs(imgs, i, span=1)

    # === === === === === Refine === === === === === === ===
    # Edge detection
    conv = convolve2d(pred, kernel, mode="same")
    conv = get_binary(conv)
    # Remove noisy pixels
    for _ in range(5):
        conv = convolve2d(conv, gauss, mode="same")
        conv = get_binary(conv, cutabs=.5)
    OUT["shades"] += [pred]
    OUT["edges"]  += [conv]

    # === === === === === Clustering === === === === === ===
    try:
        clusters, centers, np_yx = do_k_means(conv, k)
        OUT["clusters"][i]       = clusters
        OUT["centers"][i, :, 0]  = centers[:, 0]  # y
        OUT["centers"][i, :, 1]  = centers[:, 1]  # x
        OUT["yx_edges"][i]       = np_yx
    except Exception as e:
        print(e)
        None

pickle.dump(OUT, open("checkpoint.h5", "wb"))
# === === === === === check point === === === === ===
import copy
import pickle
from lib import *
from scipy.signal import spline_filter
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.signal import find_peaks

path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/group"
os.chdir(path_project)


OUT = pickle.load(open("checkpoint.h5", "rb"))
# === === === === === check point === === === === ===

# parameters
n_id = 2
n_tags = 10
k = n_id * n_tags

OUT["distance"]   = np.zeros((n, k))
OUT["directions"] = np.zeros((n, k, 2))
for i in range(n):
    OUT = sort_by_dist(OUT, i)
# distance travel to i frame
OUT["distance"][1:] = OUT["distance"][:-1]



fts_dir = np.swapaxes(OUT["directions"], 1, 0).\
                reshape((k, -1)).\
                astype(np.float32)

k_to_id, _ = cv_k_means(fts_dir, n_id)
k_to_id += 1 # which k cluster -> which id
k_to_id = np.array(list(range(1, k + 1))) # DEBUG === === === === === === === === ===

sd       = np.std(OUT["distance"])
cut      = np.median(OUT["distance"]) + 2 * sd
# n by k
show_k   = OUT["distance"] <= cut


# === === === === === Finalize prediction === === === === === ===
OUT["predictions"] = np.zeros((n, ) + OUT["edges"][0].shape)

for i in range(n):
    if OUT["clusters"][i] is not None:
        clt      = OUT["clusters"][i]
        pts      = OUT["yx_edges"][i].astype(np.int)
        show     = show_k[i]
        which_id = (k_to_id * show)[clt]
        OUT["predictions"][i][pts[:, 0], pts[:, 1]] = which_id


# === === === === === Demo === === === === === ===
std = np.std(OUT["predictions"] > 0, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(std)

# Movement by time
std = np.std(OUT["predictions"][:50] > 0, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(std)
std = np.std(OUT["predictions"][50:100] > 0, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(std)
std = np.std(OUT["predictions"][100:150] > 0, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(std)
std = np.std(OUT["predictions"][250:300] > 0, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(std)

# Relative distance
dist = []
for i in range(len(OUT["centers"])):
    pts = OUT["centers"][i][[10, 15]]
    dist += [distance(pts[0], pts[1])]

plt.figure(figsize=(16, 10))
plt.plot(dist)

# Travel distance
OUT["distance"].shape
plt.figure(figsize=(16, 10))
for i in [10, 15]:
    plt.plot(OUT["distance"][10:, i], label="pig %d" % i)
plt.legend()

# cumulated movement
cum = np.cumsum(OUT["distance"][10:, [10, 15]], axis=0)
for i in range(2):
    plt.plot(cum[:, i], label="pig %d" % (i+1))
plt.legend()

cum = np.cumsum(OUT["distance"][200:300, [10, 15]], axis=0)
for i in range(2):
    plt.plot(cum[:, i], label="pig %d" % (i+1))
plt.legend()

cum = np.cumsum(OUT["distance"][100:200, [10, 15]], axis=0)
for i in range(2):
    plt.plot(cum[:, i], label="pig %d" % (i+1))
plt.legend()
cum = np.cumsum(OUT["distance"][:100, [10, 15]], axis=0)
for i in range(2):
    plt.plot(cum[:, i], label="pig %d" % (i+1))
plt.legend()


OUT["distance"].shape


np_sd1 = np.std(OUT["predictions"]==5, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(np_sd1)

np_sd2 = np.std(OUT["predictions"]==15, axis=0)
plt.figure(figsize=(16, 10))
plt.imshow(np_sd2)

cc = copy.deepcopy(OUT["predictions"][:10])
cc.shape
np.std(cc == 10)
np.std(cc == 9)
(cc>10)[3, :10, :10]
(cc<=10).shape

np.sum(np_sd2)

# === === === === === Demo === === === === === ===
i = 50
# Edge detection
plt.figure(figsize=(16, 10))
plt.imshow(imgs[i])
plt.figure(figsize=(16, 10))
plt.imshow(imgs[i+1])
plt.figure(figsize=(16, 10))
plt.imshow(imgs[i-1])
plt.figure(figsize=(16, 10))
plt.imshow(imgs[i+2])
plt.figure(figsize=(16, 10))
plt.imshow(imgs[i-2])


pred = detect_imgs(imgs, i, span=1)
plt.figure(figsize=(16, 10))
plt.imshow(pred)

# Edge detection
conv = convolve2d(pred, kernel, mode="same")
conv = get_binary(conv)

plt.figure(figsize=(16, 10))
plt.imshow(conv)

# Remove noisy pixels
for _ in range(5):
    conv = convolve2d(conv, gauss, mode="same")
    conv = get_binary(conv, cutabs=.5)

plt.figure(figsize=(16, 10))
plt.imshow(conv)





(3, 6) + (3,)

OUT.keys()


OUT["directions"]





xx = sort_by_dist(OUT, i)
xx




np.sum([np.sum(c) for c in cls][2:20])
np.sum([np.sum(c) for c in ls_clt][2:20])

cls[5:10]
ls_clt[5:10]






cts




plt.hist(ls_travel.reshape(-1))


ls_min


cts[:5]


cts[:7]

cts[:5]

[]

cts


i = 3
for m in range(i, i + 15):
    plt.scatter(cts[m, :, 1], cts[m, :, 0])

i = 15
plt.imshow(img_c[i])
for j in range(n_tags):
    plt.scatter(ls_yx[i][ls_clt[i] == j][:, 1], ls_yx[i][ls_clt[i] == j][:, 0])






np_cts[0]

i = 105

ls_x = []
ls_y = []
ls_k = []
ls_i = []



dt = pd.DataFrame(np_yx)
dt.loc[:, "k"] = clusters
dt.columns = ["y", "x", "k"]







i = 2
imgconv = convolve2d(pred[i], kernel, mode="same")
out     = get_binary(imgconv)
plt.imshow(out)

i = 3
imgconv = convolve2d(pred[i], kernel, mode="same")
out     = get_binary(imgconv)
plt.imshow(out)

plt.imshow(pred[i])



pred[64][:30, 250:270]
np.sum(pred[20])
np.sum(pred[64])

plt.imshow(pred[20])

plt.imshow(pred[15])
plt.imshow(pred[62])
plt.imshow(pred[2])
plt.imshow(pred[6])
plt.imshow(pred[10])
plt.imshow(pred[15])
plt.imshow(pred[20])
plt.imshow(pred[64])
plt.plot(pred)


# smooth
burnin = 16
cut_sd = 2
n_smth = 5

signals     = cx.copy()
signals_sm  = smooth_signals(signals, n=n_smth)
signals_abs = np.abs(signals - signals_sm)
signals_fx  = np.array(signals.copy())


std = np.std(signals_abs[burnin:])
is_out = signals_abs > np.median(signals_abs[burnin:]) + std * cut_sd
is_out[:burnin] = False
idx_out = np.where(is_out)[0]

signals_fx[idx_out] = signals_sm[idx_out]


plt.plot(signals)
plt.plot(signals_fx)
plt.plot(signals_sm)
plt.plot(signals_abs)




# diff
burnin  = 5
signals = np.array(cx.copy())
cut_sd  = 2
diff    = np.abs(np.diff(signals[burnin:]))
std     = np.std(diff)
is_out  = diff >= (std * cut_sd)
idx_out = np.where(is_out)[0] + burnin - 1# next frame is outliers
signals_sm = smooth_signals(signals, n=5)

len(idx_out)

plt.plot(signals[burnin+1:])
plt.plot(signals_sm[burnin+1:])

plt.plot(diff)



signals_f  = signals.copy()
signals_sm = smooth_signals(signals, n=5)
signals_f[idx_out] = signals_sm[idx_out]
signals_f[idx_out+1] = signals_sm[idx_out+1]
signals_f[idx_out-1] = signals_sm[idx_out-1]

plt.plot(signals)
plt.plot(signals_f)
plt.plot(signals_sm)

plt.plot(signals[idx_out])


# find peaks

burnin = 5
signals = np.array(cx.copy())
cut_sd = 3
peaks, _ = find_peaks(signals, prominence=100)
signals_sm = smooth_signals(signals, n=5)
signals_f = signals.copy()
signals_f[peaks] = signals_sm[peaks]

plt.plot(signals[peaks])
plt.plot(signals_f[peaks])
plt.plot(signals_sm[peaks])

plt.plot(signals[150:250])
plt.plot(signals_f[150:250])




plt.plot(signals)
plt.plot(peaks, signals[peaks], "x")
plt.plot(np.zeros_like(signals), "--", color="gray")
plt.show()

std = np.std(diff)
is_out = diff >= (std * cut_sd)
idx_out = np.where(is_out)[0] + burnin - 1  # next frame is outliers

signals_f = signals.copy()
signals_sm = smooth_signals(signals, n=5)
signals_f[idx_out] = signals_sm[idx_out]
signals_f[idx_out+1] = signals_sm[idx_out+1]
signals_f[idx_out-1] = signals_sm[idx_out-1]

plt.plot(signals)
plt.plot(signals_f)
plt.plot(signals_sm)

plt.plot(signals[idx_out])






n_objs = 1
rate   = 5
k   = n_objs * rate

i = 10
img = pred[i]
dt  = get_k_centers(img, k)
img = pred[i + 1]
dt2 = get_k_centers(img, k)
img = pred[i + 2]
dt3 = get_k_centers(img, k)

plt.scatter(dt.iloc[:, 2], dt.iloc[:, 1])
plt.scatter(dt2.iloc[:, 2], dt2.iloc[:, 1])
plt.scatter(dt3.iloc[:, 2], dt3.iloc[:, 1])

