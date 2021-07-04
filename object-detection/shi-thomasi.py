import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)

flat_chess = cv2.imread("../data/flat_chessboard.png")
real_chess = cv2.imread("../data/real_chessboard.jpg")

flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_RGB2GRAY)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)

corners = cv2.goodFeaturesToTrack(gray_flat_chess, 5, 0.01, 10)

corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x, y), 5, (255, 0, 0), -1)

# plt.imshow(flat_chess)

crnrs = cv2.goodFeaturesToTrack(gray_real_chess, 100, 0.01, 10)
crnrs = np.int0(crnrs)
for i in crnrs:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 4, (255, 0, 1), -1)

# plt.imshow(real_chess)

real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2BGR)
while True:
    cv2.imshow("chess", real_chess)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
