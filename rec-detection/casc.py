import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

nadia = cv2.imread("../data/Nadia_Murad.jpg", 0)

denis = cv2.imread("../data/Denis_Mukwege.jpg", 0)

solvay = cv2.imread("../data/solvay_conference.jpg")

face_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_default.xml")


def detect_face(img):
    # time.sleep(0.01)
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, minNeighbors=4, scaleFactor=1.2)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=4)

    # display(face_img)

    return face_img


while True:
    ret, frame = cap.read()

    face = detect_face(frame)
    cv2.imshow("frame", face)

    key = cv2.waitKey(10)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
