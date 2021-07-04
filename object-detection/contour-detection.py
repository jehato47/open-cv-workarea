import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

plt.interactive(True)

img = cv2.imread("../data/internal_external.png", 0)

contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(img.shape)

for i in range(len(hierarchy[0])):
    # External

    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, 10)

internal_contours = np.zeros(img.shape)

for i in range(len(hierarchy[0])):
    # Internal

    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)

plt.imshow(internal_contours, cmap="gray")
