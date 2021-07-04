import cv2
import numpy as np
import matplotlib.pyplot as plt

# plt.interactive(True)

full = cv2.imread("../data/sammy.jpg")
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

face = cv2.imread("../data/sammy_face.jpg")
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']

# full_copy = full.copy()
for m in methods:
    full_copy = full.copy()
    res = cv2.matchTemplate(full_copy, face, eval(m))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val)

    if eval(m) in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topleft = min_loc  # (x, y)
    else:
        topleft = max_loc

    heigth, width, channels = face.shape

    bottom_right = (topleft[0] + width, topleft[1] + heigth)

    cv2.rectangle(full_copy, topleft, bottom_right, 255, thickness=4)

    plt.subplot(121)
    plt.imshow(res)
    plt.subplot(122)
    plt.imshow(full_copy)

    plt.suptitle(m)
    plt.show()
