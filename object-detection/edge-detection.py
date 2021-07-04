import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)
img = cv2.imread("../data/sammy_face.jpg")

med_val = np.median(img)
lower = int(max(0, 0.7 * med_val))
upper = int(min(255, 1.3 * med_val))

# edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)

blurred = cv2.blur(img, ksize=(7, 7))
edges = cv2.Canny(image=blurred, threshold1=lower, threshold2=upper)

plt.imshow(edges)
