import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)


def display(imgg):
    plt.imshow(imgg, cmap="gray")


reeses = cv2.imread("data/reeses_puffs.png")
cereals = cv2.imread("data/many_cereals.jpg")

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

index_params = dict(alghoritm=0, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

good = []

for i, (match1, match2) in enumerate(matches):
    if)