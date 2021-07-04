import cv2

l_clicked = False
ct = (0, 0)


def draw_circle(event, x, y, flags, params):
    global l_clicked, ct
    if event == cv2.EVENT_LBUTTONDOWN:
        ct = (x, y)
        l_clicked = True
    # else:
    #     l_clicked = False


frame = None
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow("circle")
cv2.setMouseCallback("circle", draw_circle)

while True:
    ret, frame = cap.read()
    if l_clicked:
        cv2.circle(frame, center=ct, radius=50, color=(255, 0, 0), thickness=5)
    cv2.imshow("circle", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
