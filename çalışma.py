import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)

plate_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_russian_plate_number.xml")

img = cv2.imread("C:/Users/LENOVO/Desktop/qqq.jfif")

rects = plate_cascade.detectMultiScale(img, scaleFactor=1, minNeighbors=4)

for x, y, w, h in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=10)
    print(x)

plt.imshow(img)
