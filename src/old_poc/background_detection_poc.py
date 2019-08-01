import numpy as np
import cv2 as cv
cap = cv.VideoCapture('data/dynamic_sample.mp4')
#fgbg = cv.createBackgroundSubtractorMOG2()
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    # gray = cv.cvtColor(fgmask,cv.COLOR_GRAY2BGR)
    # edges = cv.Canny(gray,50,150,apertureSize = 3)
    #
    # lines = cv.HoughLines(fgmask,500,np.pi/180,200)
    # for line in lines:
    #     for rho,theta in line:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))
    #
    #         cv.line(gray,(x1,y1),(x2,y2),(0,0,255),2)

    cv.imshow('frame',fgmask)
    #cv.imshow('frame', gray)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
