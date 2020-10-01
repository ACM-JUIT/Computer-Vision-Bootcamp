import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.GaussianBlur(img, (5,5), 0)

    canny = cv2.Canny(blur_img, 100, 150)
    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)
    
    cv2.imshow("Real", img)
    cv2.imshow("Canny Edge", canny)
    cv2.imshow("Laplacian Edge", laplacian)

    if (cv2.waitKey(1) & 0XFF == ord('q')):
        break
    
cam.release()
cv2.destroyAllWindows()
