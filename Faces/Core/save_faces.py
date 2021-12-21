import cv2 as cv
import imutils
import numpy as np

cap = cv.VideoCapture(0)
detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

cont = 0
while cont < 300:
    #frame = cv.rotate(cap.read()[1], 1)
    frame = cap.read()[1]
    frame = imutils.resize(frame, width=500)
    faces = detector.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60))
    if not np.any(faces):
        continue
    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]
    cv.imwrite(f"Faces/faces/rafael/{cont}.png", face)
    cont += 1
    print("save - ", cont)
    cv.imshow("Video", face)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
