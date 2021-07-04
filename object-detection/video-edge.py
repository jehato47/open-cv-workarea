import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    med_val = np.median(frame)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    blurred = cv2.blur(frame, ksize=(5, 5))
    edges = cv2.Canny(image=frame, threshold1=lower, threshold2=upper)

    cv2.imshow("vid", edges)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
