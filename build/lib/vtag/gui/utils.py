import os

def ls_files(path=".", word_search=".png"):
    ls_imgs = os.listdir(path)
    ls_imgs.sort()
    return [f for f in ls_imgs if word_search in f]

def drawCross(x, y, painter, size=2):
    l1_st_x, l1_st_y = x-size, y-size
    l1_ed_x, l1_ed_y = x+size, y+size
    l2_st_x, l2_st_y = x-size, y+size
    l2_ed_x, l2_ed_y = x+size, y-size
    painter.drawLine(l1_st_x, l1_st_y, l1_ed_x, l1_ed_y)
    painter.drawLine(l2_st_x, l2_st_y, l2_ed_x, l2_ed_y)

