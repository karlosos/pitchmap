"""
File with masks
"""
import cv2
import numpy as np


def grass(frame):
    """
    Delete non-green elements from frame.
    :param frame:
    :return: frame with black areas where there's no grass
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([75, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Mask editing - morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, kernel, iterations=7)
    mask = cv2.erode(mask, kernel, iterations=7)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    return res
