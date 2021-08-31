import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("> PATH TO YOUR VIDEO HERE <")
detector = pm.poseDetector()

count = 0
# 0, going up
# 1, going down
dir = 0

pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (896, 672))

    # img = cv2.imread("PoseImages/workout1.jpg")
    # img = cv2.resize(img, (480, 640))

    img = detector.findPose(img, False)

    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        # right arm
        # detector.findAngle(img, 12, 14, 16)
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)

        per = np.interp(angle, (210, 310), (0, 100))

        bar = np.interp(angle, (210, 310), (560, 140))
        # print(angle, per)

        # check for the dumbell curls
        # rgb(253, 149, 0)
        color = (0, 149, 253)
        if per == 100:
            # #e351e0
            #rgb(227, 81, 224)
            # rgb(255, 255, 1)
            color = (1, 255, 255)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            # rgb(154, 125, 2)
            color = (2, 125, 154)
            # color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # draw bar
        cv2.rectangle(img, (770, 140), (840, 560), 
            color, 3)
        cv2.rectangle(img, (770, int(bar)), (840, 560), 
            color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (777, 140), 
            cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # curl count
        cv2.rectangle(img, (10, 370), (130, 470), (31, 31, 31), cv2.FILLED)
        cv2.putText(img, str(int(count)), (15, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (110, 206, 231), 5)

    # fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), 
        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
