import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

x = width // 2
y = heigth // 2

w = width // 4
h = heigth // 4

print(width, heigth)

while True:
    ret, frame = cap.read()

    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=4)

    cv2.imshow("videoo", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
