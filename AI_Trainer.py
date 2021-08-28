import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("PoseVideos/7.mp4")
detector = pm.poseDetector()

while True:
    # success, img = cap.read()
    # img = cv2.resize(img, (640, 480))

    img = cv2.imread("PoseImages/workout1.jpg")
    img = cv2.resize(img, (480, 640))

    img = detector.findPose(img, False)

    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        detector.findAngle(img, 12, 14, 16)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
