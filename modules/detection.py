#!/usr/bin/env python

#import numpy as np
import cv2
import numpy as np
import threading
from os import getcwd

from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D

def predictWithModel(model,frame):

  x=reshape(frame) #sliding window or scaling
  labels=model.predict_on_batch(x)
  print(labels)
  return lbl, loc
  #row/col conversion
def markDetected(mat,loc):
  # Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])  None
  pass

def onlinePrediction(model,source=0, show_source=True, output='output.avi'):
  cap = cv2.VideoCapture(source)
  h,w=int(cap.get(3)),int(cap.get(4))
  #source != 0
  if source:
    fps=cap.get(7)
  else:
    fps=30

  T=1000//fps if fps else 25
  fourcc = cv2.VideoWriter_fourcc(*'X264')# I want .h264. Maybe parameterize
  out = cv2.VideoWriter(output,fourcc, fps, (h,w)) #Thread 3, #rename out,output
  #thread inits?
  cv2.namedWindow('Predictor Output')
  cv2.namedWindow('Source Video')
  cv2.moveWindow('Predictor Output', 0, 0)

  #You didnt think of multiple detections!
  while cap.isOpened():
    # Capture frame-by-frame
    ret, rawframe = cap.read()
    if ret:
      #print(listdir('.'))
      #print(ret)
      # Thread 1:
      #lbl, loc=predictWithModel(rawframe)
      lbl=False
      if lbl:
        processed=markDetected(rawframe,loc)
      else: #potentiall combine these  in mark detected.
        #processed=[]
        processed=np.copy(rawframe) ### THIS IS NOT HIS FINAL FORM. ineffective
        cv2.putText(processed, 'X',(25,85), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,255))

      cv2.imshow('Predictor Output',processed)
      # Thread 2:
      cv2.imshow('Source Video',rawframe)
      # Thread 3? (or 1):
      out.write(processed)

      if cv2.waitKey(int(25)) & 0xFF == ord('q'):
        break
    else:
      break

  # When everything done, release the capture
  cap.release()
  out.release()
  cv2.destroyAllWindows()

##    cv2.moveWindow

def onlineMultiModelPrediciton():
  #thinking of using multiple predictors to test at once.
  pass

def main():
  vid='berlin1.mp4'
  onlinePrediction(0,0)

if __name__ == "__main__":
      exit(main())
