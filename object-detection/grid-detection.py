import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)

flat_chess = cv2.imread("../data/flat_chessboard.png")

found, corners = cv2.findChessboardCorners(flat_chess, (7, 7), )

cv2.drawChessboardCorners(flat_chess, (7, 7), corners, found)

# plt.imshow(flat_chess)


dots = cv2.imread("../data/dot_grid.png")

found, corners = cv2.findCirclesGrid(dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(dots, (10, 10), corners, found)
plt.imshow(dots)
