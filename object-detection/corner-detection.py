import cv2
import matplotlib.pyplot as plt
import numpy as np

flat_chess = cv2.imread("../data/flat_chessboard.png")
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
# gray = np.float32(gray_flat_chess)

real_chess = cv2.imread("../data/real_chessboard.jpg")
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)
plt.interactive(True)

# plt.imshow(gray_flat_chess, cmap="gray")
real_chess[dst > dst.max() * 0.01] = [255, 0, 0]
plt.imshow(real_chess, cmap="gray")
