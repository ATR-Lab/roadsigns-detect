import cv2
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import time
import vlc
import random

print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=0).start()
time.sleep(2.0)

while True:
  frame = vs.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  detector = cv2.CascadeClassifier("./res/Stopsign_HAAR_19Stages.xml")
  rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

  for (x, y, w, h) in rects:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  frame = imutils.resize(frame, width=400)
  cv2.imshow("Frame", frame)
  cv2.waitKey(1)