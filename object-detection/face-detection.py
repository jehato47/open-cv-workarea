import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(True)


def display(img):
    plt.imshow(img, cmap="gray")


nadia = cv2.imread("../data/Nadia_Murad.jpg", 0)

denis = cv2.imread("../data/Denis_Mukwege.jpg", 0)

solvay = cv2.imread("../data/solvay_conference.jpg")

face_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_default.xml")


def detect_face(img):
    time.sleep(0.01)
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=4)

    # display(face_img)

    return face_img


def adj_detect_face(img):
    time.sleep(0.01)
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=4)

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    display(face_img)

    # return face_img


eye_cascade = cv2.CascadeClassifier("../data/haarcascades/haarcascade_eye.xml")

eye_rects = None


def detect_eye(img):
    global eye_rects
    time.sleep(0.01)
    eye_img = img.copy()

    eye_rects = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in eye_rects:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=4)

    display(eye_img)

    # return eye_img


detect_eye(nadia)
