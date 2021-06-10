from lib import *
from scipy.signal import spline_filter
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.signal import find_peaks

path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/One_Pig"
os.chdir(path_project)

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


k = 1
npk = 3
n_tags = k * npk

img_p = []
img_c = []
np_cts = np.zeros((n, n_tags, 2))
ls_clt = n * [None]
ls_yx  = n * [None]
for i in range(n):
# for i in range(20):
    if i % 10 == 0:
        print("%d\r" % (i))
    # Detect candidate pixels
    pred = detect_imgs(imgs, i, span=1)
    # Edge detection
    conv = convolve2d(pred, kernel, mode="same")
    conv = get_binary(conv)
    # Remove noisy pixels
    for _ in range(5):
        conv = convolve2d(conv, gauss, mode="same")
        conv = get_binary(conv, cutabs=.5)
    img_p += [pred]
    img_c += [conv]
    # Clustering
    try:
        clusters, centers, np_yx = do_k_means(conv, n_tags)
        ls_clt[i]       = clusters
        ls_yx[i]        = np_yx
        np_cts[i, :, 0] = centers[:, 0] # y
        np_cts[i, :, 1] = centers[:, 1] # x
    except Exception as e:
        print(e)
        None


img_std = np.std(imgs, axis=(0))
img_std.shape
plt.imshow(img_std)
img_p.shape


import copy

cts = np_cts.copy()
cls = copy.deepcopy(ls_clt)
ls_travel = np.zeros((n, npk))

for i in range(n):
    cts, cls, ls_travel[i] = sort_by_dist(cts, cls, i, npk)
# distance travel to i frame
ls_travel[1:] = ls_travel[:-1]

cts = np.swapaxes(cts, 1, 0).reshape((npk, -1)).astype(np.float32)
cts.shape
clusters, centers = cv_k_means(cts, k)







np.sum([np.sum(c) for c in cls][2:20])
np.sum([np.sum(c) for c in ls_clt][2:20])

cls[5:10]
ls_clt[5:10]






cts

sd       = np.std(ls_travel)
cut      = np.median(ls_travel) + 2 * sd
idx_mute = ls_travel > cut




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

