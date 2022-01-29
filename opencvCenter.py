import cv2
import numpy as np

resim = cv2.imread("soru1.jpg")
gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gri_resim, 30, 200)

contours,_ = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawContours(resim, [c], -1, (0, 255, 0), 3)
    cv2.circle(resim, (cX, cY), 7, (255, 255, 255), -1)

cv2.imshow('soru1', resim)
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('merkezleri_isaretlenmis_resim.jpg',resim)
    cv2.destroyAllWindows()