import cv2 as cv
import numpy as np
import os

recognizer=cv.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/Dell/Desktop/face detection and recognition/trainer/trainer.yml')#location of the trained model

faceCascade=cv.CascadeClassifier('C:/Users/Dell/Desktop/face detection and recognition/haarcascade_frontalface_default.xml')#haar cascade file location

id=0

font=cv.FONT_HERSHEY_SIMPLEX


name= ['none','Abhishek','David Beckham','Wolverine']

cam=cv.VideoCapture(0)

cam.set(3, 640)
cam.set(4,480)

minW= 0.1*cam.get(3)
minH=0.1*cam.get(4)

while True:
    ret,img=cam.read()

    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5, minSize= (int(minW),int(minH)),)
    
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        id, confidence= recognizer.predict(gray[y:y+h,x:x+w])
        if( confidence < 100):
            id=name[id]
            confidence=" {0}%".format(round(100-confidence))

        else:
            id="Unknown"
            confidence=" {0}%".format(round(100-confidence))
    
        cv.putText(img,str(id),(x+5,y-5),font,4,(0,255,0),1,cv.LINE_AA)
        cv.putText(img,str(confidence),(x+5,y+h-5),font,4,(0,255,0),1,cv.LINE_AA)

    cv.imshow("LIVE", img)
    k=cv.waitKey(1)
    if(k==ord('q')):
         break

cam.release()
cv.destroyAllWindows()
