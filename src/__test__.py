from lib import *

path_project = "/Users/jchen/Dropbox/projects/Virtual_Tags/data/One_Pig"
os.chdir(path_project)

paths = ls_files(path_project)
np_imgs = load_np(paths)

img_detect = detect_imgs(np_imgs, 0)

plt.imshow(img_detect)



img_detect.shape

img_detect.max()
img_detect.min()

