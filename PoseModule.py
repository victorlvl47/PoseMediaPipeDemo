import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode=False, modelComplexity=1, smooth=True, 
        segmentation=False, smoothSegmentation=True, 
        detectionCon=0.5, trackCon=0.5):
    
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, 
                self.smooth, self.segmentation, 
                self.smoothSegmentation, 
                self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, 
                    self.results.pose_landmarks, 
                    self.mpPose.POSE_CONNECTIONS)

        return img


    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList



def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()
    
    while True: 
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), 
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        imgResized = cv2.resize(img, (640, 480))

        cv2.imshow("Image", imgResized)


        cv2.waitKey(1)


if __name__ == "__main__":
    main()