import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ret, first_frame = cap.read()
first_frame = cv2.flip(first_frame, 1)
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(first_frame, dtype=np.uint8)
hsv_mask[:, :, 1] = 255

while True:
    ret1, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(first_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # TODO : Buraya bak
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)

    hsv_mask[:, :, 0] = ang / 2
    hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("o-flow", bgr)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

    first_gray = frame_gray

cap.release()
cv2.destroyAllWindows()
