import cv2
import numpy as np
import time

cap = cv2.VideoCapture("PoseVideos/7.mp4")
img = cv2.imread("PoseImages/workout1.jpg")
img = cv2.resize(img, (480, 640))

while True:
    # success, img = cap.read()
    # img = cv2.resize(img, (640, 480))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
