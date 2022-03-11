import numpy as np
from PyQt6.QtGui import QColor

colorsets = [QColor(c) for c in np.array(["#000000",
                                          "#ffff33", "#f94144", "#277da1",
                                          "#f3722c", "#f8961e", "#f9844a", "#f9c74f",
                                          "#90be6d", "#43aa8b", "#4d908e",
                                          "#577590", ])]

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