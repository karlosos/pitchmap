import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt


def grass_negative(frame, for_player=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([75, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    if for_player:
        return mask

    # Mask editing - morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, kernel, iterations=7)
    mask = cv2.erode(mask, kernel, iterations=7)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    return res, cv2.bitwise_not(mask)


def plot(movement_vectors):
    plt.style.use('ggplot')
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.array(movement_vectors)[:, 0], label="horizontal [x]")
    ax.plot(np.array(movement_vectors)[:, 1], label="vertical [y]")
    ax.set_ylabel("kierunek \n ruchu kamery")
    ax.legend()
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_ylabel("wychylenie kamery \n względem początku")
    x_cumsum = np.cumsum(np.array(movement_vectors)[:, 0])
    y_cumsum = np.cumsum(np.array(movement_vectors)[:, 1])
    ax2.plot(x_cumsum, label="horizontal [x]")
    ax2.plot(y_cumsum, label="vertical [y]")
    ax2.legend()
    plt.show()


def get_bounding_frames(cap, frame_indexes):
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    print(f"frame indexes: {frame_indexes}, total frames: {total_frames}")
    for index in frame_indexes:
        if 0 <= index <= total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            frames.append(frame)
    return frames


def main():
    cap = cv2.VideoCapture("../data/Barca_Real_continous.mp4")
    width = 100

    ret, frame1 = cap.read()
    frame1 = imutils.resize(frame1, width=width)
    _, mask = grass_negative(frame1)
    previous_frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    visualisation_frame_hsv = np.zeros_like(frame1)
    visualisation_frame_hsv[..., 1] = 255

    movement_vectors = []

    while True:
        try:
            ret, frame2 = cap.read()
            frame2 = imutils.resize(frame2, width=width)
        except AttributeError:
            break
        _, mask = grass_negative(frame2)
        next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(previous_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        movement_vectors.append(np.mean(flow, axis=(0, 1)))
        previous_frame_gray = next_frame_gray

        # Video visualisation
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        visualisation_frame_hsv[..., 0] = angle * 180 / np.pi / 2
        visualisation_frame_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        visualisation_frame_rgb = cv2.cvtColor(visualisation_frame_hsv, cv2.COLOR_HSV2BGR)
        visualisation_frame_rgb = cv2.bitwise_and(visualisation_frame_rgb, visualisation_frame_rgb, mask=mask)
        cv2.imshow('Dense OpticalFlow visualisation', visualisation_frame_rgb)

        # Inputs
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', visualisation_frame_rgb)

    cv2.destroyAllWindows()
    plot(movement_vectors)

    x_cumsum = np.cumsum(np.array(movement_vectors)[:, 0])
    y_cumsum = np.cumsum(np.array(movement_vectors)[:, 1])
    max_x = np.argmax(x_cumsum)
    min_x = np.argmin(x_cumsum)
    max_y = np.argmax(y_cumsum)
    min_y = np.argmin(y_cumsum)
    print(f"Max: {max_x}, Min: {min_x}")
    bounding_frames = get_bounding_frames(cap, (min_x, max_x, max_y, min_y))
    for idx, frame in enumerate(bounding_frames):
        #cv2.imshow(f"frame{idx}", frame)
        cv2.imwrite(f"frame{idx}.png", frame)

    cap.release()


if __name__ == '__main__':
    main()
