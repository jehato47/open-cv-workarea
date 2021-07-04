import cv2


def draw_rectangle(event, x, y, flags, params):
    global pt1, pt2, topLeftClicked, bottomRightClicked

    if event == cv2.EVENT_LBUTTONDOWN:
        if topLeftClicked and bottomRightClicked:

            topLeftClicked = False
            bottomRightClicked = False
            print(3)
            return

        elif not topLeftClicked:
            topLeftClicked = True
            pt1 = (x, y)
            print(1)

        elif not bottomRightClicked:
            bottomRightClicked = True
            pt2 = (x, y)


pt1 = (0, 0)
pt2 = (0, 0)

topLeftClicked = False
bottomRightClicked = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow("Test")
cv2.setMouseCallback("Test", draw_rectangle)

while True:
    ret, frame = cap.read()

    if topLeftClicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0, 0, 255), thickness=-1)

    # drawing rectangle
    if topLeftClicked and bottomRightClicked:
        print(122222)
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow("Test", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
