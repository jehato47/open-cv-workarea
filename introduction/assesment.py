import cv2
import matplotlib.pyplot as plt
import numpy as np

pic = cv2.imread("../../../opencv/data/dog_backpack.jpg")

fix_img = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.interactive(True)

vertices = np.array([[300, 700], [500, 400], [700, 700]])
vss = vertices.reshape(-1, 1, 2)

cv2.polylines(fix_img, [vss], color=(0, 0, 255), isClosed=True, thickness=10)
cv2.fillPoly(fix_img, [vss], (0, 0, 255))
plt.imshow(fix_img)
