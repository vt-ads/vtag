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
# app.create_labels()
# app.save_labels()
# app.save("model.h5")

edges = app.IMGS["edg"].copy()
n_blend = 2
for i in range(n_blend):
    idx = i + 1
    edges[:-idx] += edges[idx:]
    edges[idx:] += edges[:-idx]

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


# complete run
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
