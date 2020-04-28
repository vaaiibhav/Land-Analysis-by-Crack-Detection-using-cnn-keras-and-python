import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt
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
    lines = cv2.HoughLinesP(edges,1,np.pi/150,100,minLineLength,maxLineGap)
    cv2.imshow('Cracks', edges)
    
    def find_correspondence_points(img1, img2):       
##        ##SUrf starts
##        img1 = cv.imread('box.png',0)          # queryImage
##        img2 = cv.imread('box_in_scene.png',0) # trainImage
        # Initiate SIFT detector
        sift = cv.SIFT()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()
    ##fc, fc1  = find_correspondence_points(gray, "008.jpg")
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,'Crack Detected!', 
        (x2,y2), 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.imshow('cracks',frame)
    time.sleep(2)

cam.release()
cv2.destroyAllWindows()     
