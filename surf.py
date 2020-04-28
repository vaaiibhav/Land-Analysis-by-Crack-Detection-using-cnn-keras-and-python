import cv2
import numpy as np
import time

i=0
cam = cv2.VideoCapture(0)
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
    minLineLength = 5
    maxLineGap = 0.5

    cv2.imshow('Camera Input', frame)
    cv2.imshow('Gray Scale', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    lines = cv2.HoughLinesP(edges,1,np.pi/650,100,minLineLength,maxLineGap)
    cv2.imshow('Cracks', edges)
    
    def find_correspondence_points(img1, img2):
        surf = cv2.xfeatures2d.SURF_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = surf.detectAndCompute(
            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = surf.detectAndCompute(
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

        # Find point matches
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's SIFT matching ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)

        src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

        # Constrain matches to fit homography
        retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        mask = mask.ravel()

        # We select only inlier points
        pts1 = src_pts[mask == 1]
        pts2 = dst_pts[mask == 1]

        return pts1.T, pts2.T
    imagec = cv2.imread("008.jpg")
    imagecgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fc, fc1  = find_correspondence_points(gray,imagecgray )
    try:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,'Crack Detected!', 
            (x2,y2), 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.imshow('cracks',frame)
        time.sleep(3)
    except:
         print ("NO Value")
cam.release()
cv2.destroyAllWindows()



      
    


