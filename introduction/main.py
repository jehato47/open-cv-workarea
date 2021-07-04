import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# # This line sets a "seed" so you get the same random numbers we do
# np.random.seed(101)
# # This line creates an array of random numbers
# arr = np.random.randint(low=0, high=100, size=(5, 5))
#
# pic = Image.open("data/00-puppy.jpg")
#
# pic_arr = np.asarray(pic)

pic = cv2.imread("../data/00-puppy.jpg")

fix_img = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

plt.interactive(True)

