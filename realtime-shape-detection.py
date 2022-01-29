import cv2
import numpy as np

video = cv2.VideoCapture("test.mp4")

while (1):
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresold = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours :
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], 0, (0,255,0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]


    cv2.imshow("frame",frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()











