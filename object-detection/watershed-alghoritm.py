import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.interactive(True)


def display(img, cmap="gray"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)


sep_coins = cv2.imread("../data/pennies.jpg")
# sep_coins = cv2.cvtColor(sep_coins, cv2.COLOR_BGR2RGB)


blur = cv2.medianBlur(sep_coins, 25)

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

contour, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contour, i, color=(255, 0, 0), thickness=10)

display(sep_coins)
