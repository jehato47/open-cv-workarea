import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


pic = np.zeros(shape=(600, 600))

plt.interactive(True)
pic = cv2.putText(pic, "MORPH", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 25)

# plt.imshow(pic, cmap="gray")

kernel = np.ones((5, 5), dtype=np.uint8)

black_noise = np.random.randint(0, 2, (600, 600))

noise_img = pic + black_noise * 255

# noise_img[noise_img == -255] = 0
plt.imshow(noise_img, cmap="gray")
# cv2.morphologyEx()
# w_noise = np.random.randint(0, 2, (600, 600))
#
# w_noise *= 255
#
# noise_img = pic + w_noise
#
# opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
