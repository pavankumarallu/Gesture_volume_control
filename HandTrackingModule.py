import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self,mode = False,max_hands = 2,detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,self.detectionCon,self.trackCon)
        
        self.mpDraw = mp.solutions.drawing_utils
        

    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLMS,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img,handNo = 0,draw = True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            
        
            for ids,lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([ids,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,255,0),cv2.FILLED)
        
        return lmList    
        
       
    
    
def main():
    
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(False,1,0.7,0.7)

    while True:
        
        
        success, img = cap.read()
        img = detector.findHands(img)
        l = detector.findPosition(img)
        if len(l) != 0:
            print(l[8])
        cv2.imshow("Image", img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(img,str(int(fps)),(20,130),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    
        cv2.imshow("Image",img)
        k = cv2.waitKey(1)
        if k == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
    

    
    
    


if __name__ ==  "__main__":
    main()