import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(True)
img = cv2.imread("imgg.jpeg")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# cv2.imshow("i", img)

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

q = None
nonzero = None
markers = None


def find_square(image):
    global q, nonzero, markers
    markers = 0
    threshold = 10

    while np.amax(markers) == 0:
        threshold += 5
        t, img = cv2.threshold(image[:, :, 0], threshold, 255, cv2.THRESH_BINARY_INV)
        _, markers = cv2.connectedComponents(img)

    kernel = np.ones((5, 5), np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    # # q = img
    # # plt.imshow(img)
    nonzero = cv2.findNonZero(img)

    x, y, w, h = cv2.boundingRect(nonzero)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow("image", image)
    return image


a = 0
while True:

    key = cv2.waitKey(10)

    # if a == 0:
    image = find_square(img)
    cv2.imshow("image", image)

    # a += 1
    if key == ord("q"):
        break

cv2.destroyAllWindows()
