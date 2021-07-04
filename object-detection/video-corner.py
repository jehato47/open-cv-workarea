import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    time.sleep(0.1)
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 2, (255, 255, 255), thickness=2)

    cv2.imshow("vid", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
