import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import normalize


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
    print(f"1: {np.sum(hist_h)} f2: {np.sum(normalize(hist_h))}")
    hist_s = hist_s / l
    print(f"1: {np.sum(hist_s)} f2: {np.sum(normalize(hist_s))}")
    combined = hist_h.flatten().tolist() + hist_s.flatten().tolist()
    return hist_h.flatten().tolist(), hist_s.flatten().tolist(), combined


files = ['1_1', '2_3', '2_1', '3_1']
for file in files:
    frame = cv2.imread(f'data/poc/{file}.png', cv2.IMREAD_COLOR)
    mask = grass(frame)
    hist_h, hist_s, combined = extract_feature(frame)

    plt.style.use('ggplot')
    fig = plt.figure(1, figsize=(8, 5))
    gridspec.GridSpec(2, 3)

    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    cropped = cv2.bitwise_and(frame, frame, mask=mask)
    ax1.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title("obraz wejściowy")

    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax2.bar(np.arange(0, 18), hist_h, width=1)
    ax2.set_xlabel("numer przedziału")
    ax2.set_ylabel("częstotliwość")
    ax2.set_title("histogram barwy")

    ax3 = plt.subplot2grid((2, 3), (1, 1))
    plt.bar(np.arange(0, 10), hist_s, width=1)
    ax3.set_xlabel("numer przedziału")
    ax3.set_ylabel("częstotliwość")
    ax3.set_title("histogram nasycenia")

    ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax4.plot(combined)
    ax4.set_title("cecha")
    fig.tight_layout()
    plt.savefig(f"{file}.png")

    plt.show()
