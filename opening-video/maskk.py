import cv2
import numpy as np

capture = cv2.VideoCapture(0)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
out = cv2.VideoWriter('001_output.mp4', fourcc, 29.0, size, False)  # 'False' for 1-ch instead of 3-ch for color
fgbg = cv2.createBackgroundSubtractorMOG2()

while capture.isOpened():  # while Ture:
    ret, img = capture.read()
    if ret:
        fgmask = fgbg.apply(img)
        out.write(fgmask)
        cv2.imshow('img', fgmask)

    # if(cv2.waitKey(27)!=-1):  # observed it will close the imshow window immediately
    #    break                 # so change to below
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
out.release()
cv2.destroyAllWindows()
