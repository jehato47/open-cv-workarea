import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(True)

i = cv2.imread("../data/bricks.jpg").astype(np.float32) / 255

i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

img = cv2.imread("../data/sammy.jpg")
noise_img = cv2.imread("../data/sammy_noise.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(noise_img)

# cv2.putText(i, "BRICKS", (10, 600), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), thickness=5)

# i = cv2.blur(i, (10, 2))

# kernel = np.ones((5, 5), dtype=np.float32) / 25

# i = cv2.filter2D(i, -1, kernel)

# i = np.power(i, 2)

# i2 = cv2.GaussianBlur(i, (5, 5), 25)

# plt.imshow(i2)
