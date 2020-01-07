import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def grass(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([75, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    return cv2.bitwise_not(mask)


def extract_feature(frame):
    mask = grass(frame)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([frame_hsv], channels=[0], mask=mask, histSize=[18], ranges=[0, 180])
    hist_s = cv2.calcHist([frame_hsv], channels=[1], mask=mask, histSize=[10], ranges=[0, 256])
    l = np.sum(hist_h)
    hist_h = hist_h / l
    hist_s = hist_s / l
    combined = hist_h.flatten().tolist() + hist_s.flatten().tolist()
    return hist_h/l, hist_s/l, combined


frame = cv2.imread('data/poc/1_1.png', cv2.IMREAD_COLOR)
mask = grass(frame)
hist_h, hist_s, _ = extract_feature(frame)


#plt.imshow(hist, interpolation='nearest')
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(2, 2)

plt.subplot2grid((2, 2), (0, 0), rowspan=2)
cropped = cv2.bitwise_and(frame, frame,
                                      mask=mask)
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

plt.subplot2grid((2, 2), (0, 1))

plt.plot(hist_h)

plt.subplot2grid((2, 2), (1, 1))
plt.plot(hist_s)

plt.show()