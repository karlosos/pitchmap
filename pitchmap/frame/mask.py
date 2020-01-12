"""
File with masks
"""
import cv2
import numpy as np


def grass(frame, for_player=False):
    """
    Delete non-green elements from frame.
    :param frame:
    :param for_player: is it performed on player crop or entire pitch
    :return: frame with black areas where there's no grass
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('data/images/pitch_segmentation/frame.png', frame)

    # define range of blue color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([75, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # cv2.imwrite('data/images/pitch_segmentation/mask.png', mask)

    if for_player:
        return mask

    # Mask editing - morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    # cv2.imwrite('data/images/pitch_segmentation/blur.png', mask)
    mask = cv2.dilate(mask, kernel, iterations=7)
    # cv2.imwrite('data/images/pitch_segmentation/dilate.png', mask)
    mask = cv2.erode(mask, kernel, iterations=7)
    # cv2.imwrite('data/images/pitch_segmentation/erode.png', mask)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imwrite('data/images/pitch_segmentation/final.png', res)

    return res


def grass_negative(frame):
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
    res = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    return res, cv2.bitwise_not(mask)