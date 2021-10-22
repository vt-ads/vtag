path_lib = "/Users/jchen/Dropbox/projects/Virtual_Tags/src/"
os.chdir(path_lib)
from VTags import VTags
from lib import *

# arguments
dataname = "group_small"
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)


# resume
app = VTags(h5="model.h5")
# app.create_labels()
# app.save_labels()
# app.save("model.h5")
pred = app.IMGS["pred"]
k    = app.ARGS["n_id"]

n, h, w  = pred.shape
# if k = 2;
# 00 -> background
# 10 -> cluster 1
# 01 -> cluster 2
n_blend = 2
max_value = n_blend + 1
new_pred  = np.zeros((n, h, w, k))
for i in range(k):
    new_pred[pred == (i + 1), i] = 1
    print("i: ", i, " ", np.max(new_pred[59, :, :, 0]))

    for j in range(n_blend):
        idx = j + 1
        # new_pred[:-idx, :, :, i] += new_pred[idx:, :, :, i]
        new_pred[idx:, :, :, i] += (new_pred[:-idx, :, :, i] > 0)
        print("i: ", i, " j: ", j, " ", np.max(new_pred[59, :, :, 0]))
    new_pred[:, :, :, i] = new_pred[:, :, :, i] / max_value

# out_pred = new_pred.reshape((-1, k))
idx0 = np.sum(new_pred, axis=3) == 0
idx1 = new_pred[:, :, :, 0] > new_pred[:, :, :, 1]
idx2 = new_pred[:, :, :, 0] < new_pred[:, :, :, 1]




pred[idx1] = 




new_pred[:, :, :, 0] > new_pred[:, :, :, 1]


test = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [0, 2, 1],
        [0, 1, 3],
        ])
np.where(test==2)



et = out_pred[500]

ls_decode = np.array([np.where(et == max(et))[0][0] for et in out_pred])

x = [53, 67, 30]
np.where(x)



np.max(new_pred[:, :, :, 0])
np.sum(new_pred, axis=3).max()
new_pred[30, 340, 500]

# 0 1



t = 59
plt.imshow(new_pred[t, :, :, 0], cmap="gray")
plt.imshow(new_pred[t, :, :, 1], cmap="gray")
plt.imshow(pred[t], cmap="gray")


np.max(new_pred[t, :, :, 0])


new_pred[59]


new_pred.shape
pred[60].shape

pred[:, :, ]
pred
(pred == 0).shape

np.expand_dims(pred, axis=3)
new_pred.reshape(n, h, w, 1)

new_pred.shape

pred.shape

pred.shape

pred[pred!=0]
np.max(pred)





edges = app.IMGS["edg"].copy()
n_blend = 2
for i in range(n_blend):
    idx = i + 1
    edges[:-idx] += (edges[idx:]  > 0)
    edges[idx:] +=  (edges[:-idx] > 0)

edges = edges / ((n_blend * 2) + 1)


plt.imshow(app.IMGS["edg"][f], cmap="gray")


cut = np.quantile(edges[edges > 0], .5)
edges[edges >= cut] = 1
edges[edges < cut] = 0

plt.imshow(app.IMGS["edg"][f], cmap="gray")
plt.imshow(edges[f], cmap="gray")


edges.shape

edges[5:]


plt.imshow(edges[60], cmap="gray")


# complete run99
path_lib = "/Users/jchen/Dropbox/projects/Virtual_Tags/src/"
os.chdir(path_lib)
from VTags import VTags
from lib import *

# arguments
dataname = "group"
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)

app = VTags(k=2, n_tags=20)
bound_x = [180, 730, 725, 170]
bound_y = [70,  90,  460, 440]
bounds = np.array([[y, x] for x, y in zip(bound_x, bound_y)])
app.load(bounds=bounds)
app.detect_movements()
app.detect_edges()
app.detect_clusters()
app.map_k_to_id()
app.make_predictions()
app.create_labels()
app.save_labels()
app.save("model.h5")
