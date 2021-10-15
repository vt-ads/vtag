from pyqtgraph.Qt import scale
from lib import *
from VTags import VTags
# Input
dataname = "group"
# WD
path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/"
os.chdir(path_project + dataname)

# Resume run
app = VTags(k=2)
app.load(h5="model.h5")

# generate features
features = app.OUTS["features"]

# figure: feature map --- --- --- --- --- --- --- --- --- --- --- ---
i = 157
plt.imshow(features[i])

# figure: error strategies--- --- --- --- --- --- --- --- --- --- --- ---
def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

def cal_error(obs, pre):
    errors = []
    _, k = obs.shape
    for i in range(int(k / 2)):
        errors += [((pre[:, (i * 2) + 0] - obs[:, (i * 2) + 0])**2 +
                    (pre[:, (i * 2) + 1] - obs[:, (i * 2) + 1])**2) ** .5]
    return np.mean(errors, axis=0)


# get label from group data (swap 173-211)
os.chdir(path_project + "group")
pre_grp_J = np.array(pd.read_csv("labels_J1_adj.csv"))
pre_grp_F = np.array(pd.read_csv("labels.csv"))
obs_grp = np.array(pd.read_csv("truth/labels_1.csv"))

# compute the largest error
# plt.scatter(x, y)
bound = app.ARGS["bounds"]
x = bound[:, 0]
y = bound[:, 1]
base = distance([90, 730], [440, 170])

fig, axes = plt.subplots(figsize=(8, 4))
error_grpJ = (cal_error(obs_grp, pre_grp_J) / base)[2:-1]
error_grpF = (cal_error(obs_grp, pre_grp_F) / base)[2:-1]

axes.plot(error_grpJ, linewidth=2, alpha=.7)
axes.plot(error_grpF, linewidth=2, alpha=.7)
axes.yaxis.grid(True)
axes.set_xlabel('Time Frames')
axes.set_ylabel('Standardized Error')

fig, axes = plt.subplots(figsize=(5, 4))
plot_lbs = ["James", "FangYi"]
facecolor = ['#1f77b4', '#ff7f03', '#2ca02c']
axes.yaxis.grid(True)
axes.set_ylabel('Standardized Error')
plot = axes.boxplot([error_grpJ, error_grpF],
                    notch=True,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=plot_lbs)
for patch, color in zip(plot["boxes"], facecolor):
    patch.set_facecolor(color)


# figure: error --- --- --- --- --- --- --- --- --- --- --- ---
def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

def cal_error(obs, pre):
    errors = []
    _, k = obs.shape
    for _ in range(int(k)):
        errors += [((pre[:, 0] - obs[:, 0])**2 + (pre[:, 1] - obs[:, 1])**2) ** .5]
    return np.mean(errors, axis=0)


# get label from group data (swap 173-211)
os.chdir(path_project + "group")
pre_grp     = np.array(pd.read_csv("labels.csv"))
pre_grp_adj = np.array(pd.read_csv("labels_adj.csv"))
obs_grp     = np.array(pd.read_csv("truth/labels_1.csv"))

# get label from single data
os.chdir(path_project + "one_pig")
pre_one     = np.array(pd.read_csv("labels.csv"))
obs_one     = np.array(pd.read_csv("truth/labels.csv"))

# compute the largest error
# plt.scatter(x, y)
bound = app.ARGS["bounds"]
x = bound[:, 0]
y = bound[:, 1]
base = distance([90, 730], [440, 170])

fig, axes = plt.subplots(figsize=(8, 4))
error_grp = (cal_error(obs_grp, pre_grp) / base)[2:-1]
error_grpa = (cal_error(obs_grp, pre_grp_adj) / base)[2:-1]
error_one = (cal_error(obs_one, pre_one) / base)[2:-1]
axes.plot(error_grp, linewidth=2, alpha=.7)
axes.plot(error_grpa + .005, linewidth=2, alpha=.7)
axes.plot(error_one, linewidth=2, alpha=.7)
axes.yaxis.grid(True)
axes.set_xlabel('Time Frames')
axes.set_ylabel('Standardized Error')

fig, axes = plt.subplots(figsize=(5, 4))
plot_lbs = ["Two-Pig Case", "Two-Pig Case\n(Corrected)", "One-Pig Case"]
facecolor = ['#1f77b4', '#ff7f03', '#2ca02c']
axes.yaxis.grid(True)
axes.set_ylabel('Standardized Error')

plot = axes.boxplot([error_grp, error_grpa, error_one],
             notch=True,
             vert=True,  # vertical box alignment
             patch_artist=True,  # fill with color
             labels=plot_lbs)
for patch, color in zip(plot["boxes"], facecolor):
    patch.set_facecolor(color)

err = error_one
np.mean(err > (np.median(err) +
               1.5 * (np.quantile(err, .75) -
                      np.quantile(err, .25))))

# figure: movement --- --- --- --- --- --- --- --- --- --- --- ---
os.chdir(path_project + "group")
pre_grp = np.array(pd.read_csv("labels.csv")).reshape((300, 2, 2))

dist = np.array([distance(p1, p2) for p1, p2 in pre_grp])

fig, axes = plt.subplots(figsize=(14, 6))
axes.plot(dist[2:-1]/base)
axes.yaxis.grid(True)
axes.set_xlabel('Timeframe')
axes.set_ylabel('Distance Between Pigs (Pixel)')


# figure: spatial --- --- --- --- --- --- --- --- --- --- --- ---
# two-pig
os.chdir(path_project + "group")
app = VTags()
app.load(h5="model.h5")
pred = app.IMGS["pred"]
std_1 = (pred == 1).std(axis=0)
std_2 = (pred == 2).std(axis=0)

plt.figure(figsize=(14, 10))
plt.imshow(std_1)
plt.figure(figsize=(14, 10))
plt.imshow(std_2)

mov = app.IMGS["mov"]
std = (mov > 1).std(axis=0)
plt.figure(figsize=(14, 10))
plt.imshow(std)

# one-pig
os.chdir(path_project + "one_pig")
app = VTags()
app.load(h5="model.h5")
mov = app.IMGS["mov"]
std = (mov > 0).std(axis=0)
plt.figure(figsize=(14, 10))
plt.imshow(std)

# figure: active --- --- --- --- --- --- --- --- --- --- --- ---
os.chdir(path_project + "group")
app = VTags()
app.load(h5="model.h5")
lbs = app.OUTS["labels"]

motion = lbs[3:-1] - lbs[2:-2]
motion_1 = motion[:, 0]
motion_2 = motion[:, 1]
mov_1 = np.array([distance(m, (0, 0)) for m in motion_1])
mov_2 = np.array([distance(m, (0, 0)) for m in motion_2])

fig, axes = plt.subplots(figsize=(8, 6), nrows=2)
axes[0].plot(mov_1, alpha=.7)
axes[0].plot(mov_2, alpha=.7)
axes[0].yaxis.grid(True)
# axes[0].set_xlabel('Timeframe')
axes[0].set_ylabel('Movement (Pixel)')

axes[1].plot(np.cumsum(mov_1))
axes[1].plot(np.cumsum(mov_2))
axes[1].yaxis.grid(True)
axes[1].set_xlabel('Time Frames')
axes[1].set_ylabel('Cumulated Movement (Pixel)')


fig, axes = plt.subplots(figsize=(4, 8), nrows=2)
plot = axes[0].boxplot([mov_1, mov_2],
                    notch=True,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=["Pig 1", "Pig 2"])
facecolor = ['#1f77b4', '#ff7f03', '#2ca02c']
axes[0].yaxis.grid(True)
axes[0].set_ylabel('Movement (Pixel)')
for patch, color in zip(plot["boxes"], facecolor):
    patch.set_facecolor(color)

idx = (mov_1 > 0)
axes[1].scatter(mov_1[idx], mov_2[idx], alpha=.5)
# axes[1].scatter(mov_1[~idx], mov_2[~idx], alpha=.5)
axes[1].xaxis.grid(True)
axes[1].yaxis.grid(True)
axes[1].set_ylabel('Movements of Pig 2 (pixel)')
axes[1].set_xlabel('Movements of Pig 1 (pixel)')

pearsonr(mov_1[idx], mov_2[idx])


# figure: 3 replicates --- --- --- --- --- --- --- --- --- --- --- ---
plt.figure(figsize=(15, 6))
obs = np.array(pd.read_csv("truth/labels_1.csv"))
error = cal_error(obs, pre_grp)
plt.plot(error)

obs = np.array(pd.read_csv("truth/labels_2.csv"))
error = cal_error(obs, pre_grp)
plt.plot(error)

obs = np.array(pd.read_csv("truth/labels_3.csv"))
error = cal_error(obs, pre_grp)
plt.plot(error)

