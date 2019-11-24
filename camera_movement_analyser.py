import cv2
import numpy as np
import imutils
import mask
import frame_loader
import matplotlib.pyplot as plt


class CameraMovementAnalyser:
    def __init__(self, fl):
        self.__fl = fl
        self.__frame_width = 100

        self.x_cumsum = None
        self.x_min = None
        self.x_max = None
        self.arg_x_min = None
        self.arg_x_max = None

    def calculate_characteristic_points(self):
        self.__fl.set_current_frame_position(0)
        frame1 = self.__fl.load_frame()
        frame1 = imutils.resize(frame1, width=self.__frame_width)

        _, mask_frame = mask.grass_negative(frame1)
        previous_frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        visualisation_frame_hsv = np.zeros_like(frame1)
        visualisation_frame_hsv[..., 1] = 255

        movement_vectors = []

        while True:
            try:
                frame2 = self.__fl.load_frame()
                frame2 = imutils.resize(frame2, width=self.__frame_width)
            except AttributeError:
                break
            _, mask_frame = mask.grass_negative(frame2)
            next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(previous_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            movement_vectors.append(np.mean(flow, axis=(0, 1)))
            previous_frame_gray = next_frame_gray

            # Video visualisation
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            visualisation_frame_hsv[..., 0] = angle * 180 / np.pi / 2
            visualisation_frame_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            visualisation_frame_rgb = cv2.cvtColor(visualisation_frame_hsv, cv2.COLOR_HSV2BGR)
            visualisation_frame_rgb = cv2.bitwise_and(visualisation_frame_rgb, visualisation_frame_rgb, mask=mask_frame)
            cv2.imshow('Dense OpticalFlow visualisation', visualisation_frame_rgb)

            # Inputs
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame2)
                cv2.imwrite('opticalhsv.png', visualisation_frame_rgb)

        cv2.destroyAllWindows()
        self.plot(movement_vectors)

        x_cumsum = np.cumsum(np.array(movement_vectors)[:, 0])
        max_x = np.amax(x_cumsum)
        min_x = np.amin(x_cumsum)
        length_x = np.abs(max_x - min_x)
        x_cumsum = x_cumsum * (180/length_x)
        self.x_cumsum = x_cumsum

        y_cumsum = np.cumsum(np.array(movement_vectors)[:, 1])
        argmax_x = np.argmax(x_cumsum)
        self.arg_x_max = argmax_x
        argmin_x = np.argmin(x_cumsum)
        self.arg_x_min = argmin_x
        max_x = np.amax(x_cumsum)
        self.x_max = max_x
        min_x = np.amin(x_cumsum)
        self.x_min = min_x

        argmax_y = np.argmax(y_cumsum)
        argmin_y = np.argmin(y_cumsum)
        print(f"Max: {argmax_x}, Min: {argmin_x}")

        # bounding_frames = self.__fl.get_frames((argmin_x, argmax_x))
        # for idx, frame in enumerate(bounding_frames):
        #     cv2.imwrite(f"frame{idx}.png", frame)

        self.__fl.set_current_frame_position(0)
        return argmin_x, argmax_x, max_x, min_x

    @staticmethod
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
