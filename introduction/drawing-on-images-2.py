import cv2
import numpy as np
import matplotlib.pyplot as plt

blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)

plt.interactive(True)

cv2.rectangle(blank_img, pt1=(100, 400), pt2=(300, 200), color=(255, 0, 0), thickness=5)

cv2.circle(blank_img, center=(412, 112), color=(255, 0, 255), radius=100, thickness=5)

cv2.circle(img=blank_img, center=(400, 400), radius=50, color=(255, 0, 0), thickness=-1)

cv2.line(blank_img, pt1=(0, 0), pt2=(512, 512), color=(102, 255, 255), thickness=5, lineType=cv2.LINE_AA)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(blank_img, text="Hello", org=(10, 500), fontFace=font, fontScale=4, color=(255, 255, 255), thickness=3,
            lineType=cv2.LINE_AA)

b_img = np.zeros(shape=(512, 512, 3), dtype=np.int32)

vertices = np.array([[100, 300], [200, 200], [400, 300], [200, 400]], np.int32)

pts = vertices.reshape((-1, 1, 2))

cv2.polylines(b_img, [pts], isClosed=True, color=(255, 0, 0), thickness=5)

cv2.putText(b_img, text="polylines", fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=(255, 100, 255), thickness=5,
            org=(0, 480))

plt.imshow(b_img)
