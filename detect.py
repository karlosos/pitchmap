"""
Detects things from frame

https://github.com/arunponnusamy/cvlib
"""
import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
import numpy as np

import mask


def players_detection(frame):
    bbox, label, conf = cv.detect_common_objects(frame)
    out = draw_bbox(frame, bbox, label, conf)
    return out, bbox, label


def edges_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # edge detection
    low_threshold = 10
    high_threshold = 100
    frame_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    return frame_edges


def lines_detection(frame_edges, img):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(frame_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return line_image
