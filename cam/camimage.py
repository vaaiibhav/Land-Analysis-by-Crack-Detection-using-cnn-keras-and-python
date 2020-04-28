import numpy as np
import cv2
import time

i=0
cam = cv2.VideoCapture(1)
print (cam.isOpened())
##cam.release()
##cv2.destroyAllWindows()
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2

while(True):
   
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,460,470,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 5

    cv2.imshow('Camera Input', frame)
    cv2.imshow('Gray Scale', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    cv2.imshow('Cracks', edges)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,'Crack Detected!', 
        (x2,y2), 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.imshow('cracks',frame)
    time.sleep(5)

cam.release()
cv2.destroyAllWindows()
