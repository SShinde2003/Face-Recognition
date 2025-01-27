import numpy as np
import cv2 as cv

cap=cv.VideoCapture(0)

face_cascade=cv.CascadeClassifier("D:\face\LBPH-Face-Recognizer-main\haarcascade_frontalface_default.xml")
font=cv.FONT_HERSHEY_SIMPLEX
text=""
c=1

while True:
    ret,img=cap.read()

    if ret==True:
        
        
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, 1.3,5)
        text=str(len(faces))
        for x,y,w,h in faces:
            
            X=(h//2)+x
            Y=(w//2)+y
            img=cv.circle(img,(X,Y),(w//2),(0,0,255),(2))
            cv.putText(img,text,(500,450),font,4,(0,255,0),1,cv.LINE_AA)
            newimg= img[y:y+h,x:x+w]
            cv.imshow("Cropped",newimg)
        cv.imshow("Image",img)
        cv.imshow(" Gray Image",gray)
        k=cv.waitKey(1)

        if k== ord('q'):
            break
        if k==ord('c'):
                for i in range(30):
                    ret,img=cap.read()

                    if ret==True:
                        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                        faces=face_cascade.detectMultiScale(gray, 1.3,5)
                        for x,y,w,h in faces:
                            newimg=img[y:y+h,x:x+w]
                            imgname="D:\face\LBPH-Face-Recognizer-main\dataset"+str(c)+".jpg" #Enter the path of the folder you want to save the images to.The images should be saved in the same format, in a single folder.
                            cv.imwrite(imgname,newimg)
                            c+=1
                        k=cv.waitKey(1)
                        if k==ord('q'):
                            break



        
    else:
        break

cap.release()
cv.destroyAllWindows()
