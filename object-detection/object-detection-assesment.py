import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)
img = cv2.imread("../data/car_plate.jpg")

plate_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_russian_plate_number.xml")

plate_rectangles = plate_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4)

x, y, w, h = 0, 0, 0, 0

for x, y, w, h in plate_rectangles:
    pass

plate = img[y:y + h, x:x + w]

blurred = cv2.medianBlur(plate, 7)

img[y:y + h, x:x + w] = blurred
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
