import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)


def show_img(img):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")


pic = cv2.imread("../data/rainbow.jpg", 0)

csword = cv2.imread("../data/crossword.jpg", 0)

ret, thresh = cv2.threshold(csword, 127, 255, cv2.THRESH_TOZERO)

q = cv2.adaptiveThreshold(csword, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=5, C=8)

show_img(q)
