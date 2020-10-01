import cv2
import numpy as np

FACE_CLASSIFIER_PATH = "classifier/haarcascade_frontalface_default.xml"

#Creating an empty function
def nothing(x):
    pass

#Reading frames from webcam
cap = cv2.VideoCapture(0)

#Loading Cascade Classifier
face_cascade = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)

#Creating trackbar window 
cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame",  5,  20, nothing)

while True:

    #Grabbing each frame from video
    _, frame = cap.read()

    #Converting frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Getting number of neighbours from trackbar
    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

    #Grabbing list of faces from given frame
    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)

    #Looping through each face (region of interest) and drawing rectangle around
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    #Escape key to break loop
    key = cv2.waitKey(1)
    if key == 27:
        break

#Cleaning up
cap.release()
cv2.destroyAllWindows()
