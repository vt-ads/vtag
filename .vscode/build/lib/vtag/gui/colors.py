import numpy as np
from PyQt6.QtGui import QColor

ls_colors = np.array(["#000000",
                      "#E53D00", "#05668D", "#254D32",
                      "#34113F", "#62BBC1", "#A5BE00",
                      "#EF6461", "#BCF4F5", "#E9D758"])

def vtcolor(i, alpha=255):
    color_out = QColor(ls_colors[i])
    color_out.setAlpha(alpha)
    return color_out


alpha = 200
palette_viridis = [
    QColor(68, 1, 84, alpha),
    QColor(72, 40, 62, alpha),
    QColor(62, 74, 137, alpha),
    QColor(49, 104, 142, alpha),
    QColor(38, 130, 142, alpha),
    QColor(31, 158, 137, alpha),
    QColor(53, 183, 121, alpha),
    QColor(109, 205, 89, alpha),
    QColor(180, 222, 44, alpha),
    QColor(253, 231, 37, alpha)
]
palette_viridis.reverse()

# colormap = [qRgb(0, 0, 0),       # 0
#             qRgb(255, 255, 51),  # 1
#             qRgb(55, 126, 184),  # 2
#             qRgb(77, 175, 74),   # 3
#             qRgb(228, 26, 28),   # 4
#             qRgb(152, 78, 163),  # 5
#             qRgb(255, 127, 0),   # 6
#             qRgb(13, 136, 250),  # 7
#             qRgb(247, 129, 191),  # 8
#             qRgb(153, 153, 153)]  # 9

# "#f3722c", "#f8961e", "#f9844a", "#f9c74f",
# "#90be6d", "#43aa8b", "#4d908e",
# "#577590"