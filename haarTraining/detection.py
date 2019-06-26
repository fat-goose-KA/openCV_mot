import numpy as np
import cv2
moth_cascade = cv2.CascadeClassifier('haarTraining_output.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
moth = moth_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in moth:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
   # eyes = eye_cascade.detectMultiScale(roi_gray)
   # for (ex,ey,ew,eh) in eyes:
   #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imwrite("0626result.jpg",img)
cv2.waitKey(0)
#cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('test2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
   # eyes = eye_cascade.detectMultiScale(roi_gray)
   # for (ex,ey,ew,eh) in eyes:
   #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imwrite("0626result2.jpg",img)
cv2.waitKey(0)
#cv2.destroyAllWindows()

