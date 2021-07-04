import cv2
import matplotlib.pyplot as plt
import numpy as np

pic = cv2.imread("C:/Users/LENOVO/opencv/data/dog_backpack.png")
pic2 = cv2.imread("C:/Users/LENOVO/opencv/data/watermark_no_copy.png")

fix_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
fix_pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)

# fix_pic = cv2.resize(fix_pic, (1200, 1200))
# fix_pic2 = cv2.resize(fix_pic2, (1200, 1200))

# new_pic = cv2.addWeighted(fix_pic, 0.5, fix_pic2, 0.5, 10)

plt.interactive(True)

# plt.imshow(new_pic)
fix_pic2 = cv2.resize(fix_pic2, (600, 600))
x_end = 2110 + fix_pic2.shape[1]
y_end = 2110 + fix_pic2.shape[0]

fix_pic[2110: y_end, 2110:x_end] = fix_pic2

plt.imshow(fix_pic)
