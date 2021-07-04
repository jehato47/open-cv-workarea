import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../data/sudoku.jpg", 0)

plt.interactive(True)


def display_image(i):
    plt.imshow(i, cmap="gray")


sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

mixed = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
gr = cv2.morphologyEx(mixed, cv2.MORPH_GRADIENT, kernel=np.ones((4, 4), dtype=np.uint8))

display_image(gr)
