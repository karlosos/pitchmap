"""
Tracking of points from cv2.goodFeaturesToTrack
"""

import numpy as np
import cv2


def setup():
    cap = cv2.VideoCapture("dynamic_sample.mp4")
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    return [cap, feature_params, lk_params, color, old_gray, mask]


def load_frame(cap):
    ret, frame = cap.read()

    return frame


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # edge detection
    low_threshold = 10
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    return edges


def line_detection(edges, img):
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
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    return lines_edges, line_image


def optical_flow(frame, feature_params, lk_params, color, old_gray, mask, p0):
    if len(frame.shape) == 2:
        frame_gray = frame
        mask = np.zeros_like(frame_gray)
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


    try:
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    except:
        print("No points found")
        good_new = None


    return mask, frame, frame_gray, good_new

def display(frame, mask):
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)


def exit_user_input():
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        return True
    else:
        return False


def update_optical_flow_previous(frame_gray, good_new):
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    return old_gray, p0


def main():
    [cap, feature_params, lk_params, color, old_gray, mask] = setup()

    img = load_frame(cap)
    edges = edge_detection(img)
    line_and_edges, lines = line_detection(edges, img)

    p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(line_and_edges, cv2.COLOR_BGR2GRAY), mask=None, **feature_params)

    cntr = 0
    while(1):
        cntr += 1
        cntr = cntr % 10

        img = load_frame(cap)
        edges = edge_detection(img)
        line_and_edges, lines = line_detection(edges, img)

        if cntr == 0:
            p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(line_and_edges, cv2.COLOR_BGR2GRAY), mask=None, **feature_params)

        mask, frame, frame_gray, good_new = optical_flow(lines, feature_params, lk_params, color, old_gray, mask, p0)

        if good_new is None:
            good_new = cv2.goodFeaturesToTrack(cv2.cvtColor(line_and_edges, cv2.COLOR_BGR2GRAY), mask=None,
                                             **feature_params)

        display(mask, frame)
        if exit_user_input():
            break

        old_gray, p0 = update_optical_flow_previous(frame_gray, good_new)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
