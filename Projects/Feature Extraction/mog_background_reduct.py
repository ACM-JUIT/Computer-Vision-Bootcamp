import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()