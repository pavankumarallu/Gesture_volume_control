import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam,hCam = 1000,500
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
cTime = 0

detector = htm.handDetector(False,1,0.8,0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
vol_bar = 400
vol_per = 0
while True:
    ret,img = cap.read()        
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw = False)
    if len(lmlist) != 0:
        x1,y1 = lmlist[4][1],lmlist[4][2]
        x2,y2 = lmlist[8][1],lmlist[8][2]
        cx,cy = (x1+x2)//2 , (y1+y2)//2
        
        
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,0),3)
        
        length = math.hypot(x2-x1,y2-y1)

        vol = np.interp(length,[50,220],[minVol,maxVol])
        vol_bar = np.interp(length,[45,200],[400,150])
        vol_per = np.interp(length,[45,200],[0,100])
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
    
    cv2.rectangle(img,(50,150),(85,400),(25,0,25),3)
    cv2.rectangle(img,(50,int(vol_bar)),(85,400),(25,0,25),cv2.FILLED)
    cv2.putText(img,str(int(vol_per))+"%",(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(25,0,25),3)

                    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,"FPS : "+str(int(fps)),(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(25,0,25),3)
    cv2.imshow("Image",img)
    k = cv2.waitKey(1)
    if k == 13:
        break
    

cap.release()
cv2.destroyAllWindows()