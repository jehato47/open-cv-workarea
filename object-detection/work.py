import cv2
import numpy as np
import matplotlib.pyplot as plt

pic = cv2.imread("../data/pexels-photo-176162.jpeg")
gray = cv2.imread("../data/pexels-photo-176162.jpeg", 0)

pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

plt.interactive(True)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

mixed = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

q = cv2.morphologyEx(mixed, cv2.MORPH_GRADIENT, kernel=np.ones((4, 4), dtype=np.uint8))

# plt.imshow(q, cmap="gray")
# q = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)

while True:
    cv2.imshow("q", q)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
