import cv2 as cv

capture = cv.VideoCapture(0)

facedetection = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret,frame = capture.read()
    faces = facedetection.detectMultiScale(frame,scaleFactor=1.10,minNeighbors=5)
    for x,y,w,h in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255),2)
    cv.imshow("Face_Detection",frame)
    stop = cv.waitKey(1)
    if stop == ord("v"):
        break
capture.release()
cv.destroyAllWindows()
 
