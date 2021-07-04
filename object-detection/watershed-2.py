import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.interactive(True)


def display(imgg, cmap="gray"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.imshow(imgg, cmap=cmap)


sep_coins = cv2.imread("../data/pennies.jpg")

img = cv2.medianBlur(sep_coins, 35)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), dtype=np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

rett, markers = cv2.connectedComponents(sure_fg)

markers += 1

markers[unknown == 255] = 0

# TODO : img nin neden olmadığına bak
markers = cv2.watershed(sep_coins, markers)

contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, color=(255, 0, 0), thickness=6)

display(sep_coins)
