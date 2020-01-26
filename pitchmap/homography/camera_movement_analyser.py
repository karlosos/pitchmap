import cv2
import numpy as np
import imutils

from pitchmap.segmentation import mask
import matplotlib.pyplot as plt


class CameraMovementAnalyser:
    def __init__(self, fl):
        self.__fl = fl
        self.__frame_width = 100

        self.x_cum_sum = None
        self.x_min = None
        self.x_max = None
        self.arg_x_min = None
        self.arg_x_max = None

        self.loader = None

    def get_characteristic_points(self):
        is_loaded = False
        if self.loader is not None:
            is_loaded = self.loader.load()
        if not is_loaded:
            self.analyse_camera_movement()
            if self.loader is not None:
                self.loader.save()

        return self.arg_x_min, self.arg_x_max, self.x_max, self.x_min

    def analyse_camera_movement(self):
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
            window_name = 'Dense OpticalFlow visualisation'
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            visualisation_frame_hsv[..., 0] = angle * 180 / np.pi / 2
            visualisation_frame_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            visualisation_frame_rgb = cv2.cvtColor(visualisation_frame_hsv, cv2.COLOR_HSV2BGR)
            visualisation_frame_rgb = cv2.bitwise_and(visualisation_frame_rgb, visualisation_frame_rgb, mask=mask_frame)
            cv2.imshow(window_name, visualisation_frame_rgb)

            # Inputs
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame2)
                cv2.imwrite('opticalhsv.png', visualisation_frame_rgb)

        cv2.destroyWindow(window_name)
        self.plot(movement_vectors)

        x_cum_sum = np.cumsum(np.array(movement_vectors)[:, 0])
        self.x_cum_sum = self.normalize_cum_sum(x_cum_sum)

        self.calculate_characteristic_points()
        self.__fl.set_current_frame_position(0)

    def calculate_characteristic_points(self):
        self.arg_x_max = np.argmax(self.x_cum_sum)
        self.arg_x_min = np.argmin(self.x_cum_sum)
        self.x_max = np.amax(self.x_cum_sum)
        self.x_min = np.amin(self.x_cum_sum)

    @staticmethod
    def normalize_cum_sum(x_cum_sum):
        max_x = np.amax(x_cum_sum)
        min_x = np.amin(x_cum_sum)
        length_x = np.abs(max_x - min_x)
        x_cum_sum = x_cum_sum * (180 / length_x)
        return x_cum_sum

    @staticmethod
    def plot(movement_vectors):
        plt.style.use('ggplot')
        ax = plt.subplot(2, 1, 1)
        ax.plot(np.array(movement_vectors)[:, 0], label="horizontal [x]")
        ax.plot(np.array(movement_vectors)[:, 1], label="vertical [y]")
        ax.set_ylabel("zwrot \n ruchu kamery")
        ax.legend()
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_ylabel("wychylenie kamery \n względem początku")
        x_cum_sum = np.cumsum(np.array(movement_vectors)[:, 0])
        y_cum_sum = np.cumsum(np.array(movement_vectors)[:, 1])
        ax2.plot(x_cum_sum, label="horizontal [x]")
        ax2.plot(y_cum_sum, label="vertical [y]")
        ax2.legend()
        plt.show()
