import cv2 
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideos/1.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 
            5, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), 
            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    imgResized = cv2.resize(img, (832, 624))
    cv2.imshow("Image", imgResized)
    cv2.waitKey(1)
