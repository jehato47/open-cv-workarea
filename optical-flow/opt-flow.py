import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

corner_track_params = dict(maxCorners=30, qualityLevel=0.3, minDistance=10, blockSize=7, mask=None)
lk_params = dict(winSize=(200, 200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret1, frame1 = cap.read()
frame1 = cv2.flip(frame1, 1)
fr1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

capturedPoints = cv2.goodFeaturesToTrack(fr1_gray, **corner_track_params)
# capturedPoints = np.int0(capturedPoints)
mask = np.zeros_like(frame1, dtype=np.uint8)

while True:
    ret2, frame2 = cap.read()
    frame2 = cv2.flip(frame2, 1)
    fr2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    nextPts, status, err = cv2.calcOpticalFlowPyrLK(fr1_gray, fr2_gray, capturedPoints, None, **lk_params)

    goodNew = nextPts[status == 1]
    goodPrev = capturedPoints[status == 1]

    for new, prev in zip(goodNew, goodPrev):
        xnew, ynew = new.ravel()
        xprev, yprev = prev.ravel()

        mask = cv2.line(mask, (int(xprev), int(yprev)), (int(xnew), int(ynew)), color=(255, 0, 0), thickness=5)

        frame2 = cv2.circle(frame2, (int(xnew), int(ynew)), 8, (0, 0, 255), -1)

    img = cv2.add(frame2, mask)
    cv2.imshow("frame2", img)

    key = cv2.waitKey(30)

    if key == ord("q"):
        break

    fr1_gray = fr2_gray.copy()
    capturedPoints = goodNew.reshape(-1, 1, 2)


cap.release()
cv2.destroyAllWindows()
