import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(True)

img = cv2.imread("../data/giraffes.jpg")
dark_img = cv2.imread("../data/giraffes.jpg", 0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ret, bin_img = cv2.threshold(dark_img, 127, 255, cv2.THRESH_BINARY)

hsvv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

kernel = np.ones((4, 4), dtype=np.float32) * 1 / 10

new = cv2.filter2D(img, -1, kernel)
# plt.imshow(new, cmap="gray")

kernel = np.ones((5, 5))

img = cv2.imread("../data/giraffes.jpg", 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# plt.imshow(sobelx, cmap="gray")

img = cv2.imread("../data/giraffes.jpg")

colors = ("b", "g", "r")

for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.show()

cv2.erode(img, np.ones((5, 5), dtype=np.uint8),)
