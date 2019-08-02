import numpy as np
import cv2 as cv2

cap = cv2.VideoCapture('data/dynamic_sample.mp4')
while(1):
    ret, frame = cap.read()

    # load gray image and add gaussian blur
    #img = cv2.imread('src.png')
    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # edge detection
    low_threshold = 10
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # houghlines to get the lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow('frame', lines_edges)
    #cv2.imshow('frame', edges)
    #cv.imshow('frame', gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
