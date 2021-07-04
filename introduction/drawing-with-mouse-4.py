import matplotlib.pyplot as plt
import numpy as np
import cv2

pic = cv2.imread("../data/00-puppy.jpg")

pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

plt.interactive(True)

