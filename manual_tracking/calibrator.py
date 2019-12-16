import numpy as np
import cv2
from pitchmap.homography import matrix_interp
import math


class Calibrator:
    def __init__(self):
        self.enabled = False
        self.points = {}

        self.current_point = None

        self.H_dictionary = {}

    def toggle_enabled(self):
        if not self.enabled:
            self.current_point = None
            self.points = {}

        self.enabled = not self.enabled
        return self.enabled

    def clear_points(self):
        self.points = {}

    def add_point_main_window(self, pos):
        if self.current_point is None:
            self.current_point = pos
            index = len(self.points) + 1
            return index
        else:
            return False

    def add_point_model_window(self, pos):
        if self.current_point is not None:
            index = len(self.points) + 1
            self.points[index] = (self.current_point, pos)
            self.current_point = None
            return index
        else:
            return False

    def get_points_count(self):
        return len(self.points)

    def can_perform_calibrate(self):
        if self.enabled and self.get_points_count() >= 4:
            return True
        return False

    def find_homography(self, frame):
        transformed_frame = None

        if self.enabled and \
                self.get_points_count() >= 4:
            original_points, model_points = zip(*self.points.values())
            original_points = np.float32(original_points)
            model_points = np.float32(model_points)
            rows, columns, channels = frame.shape

            H, _ = cv2.findHomography(original_points, model_points)
            transformed_frame = cv2.warpPerspective(frame, H, (columns, rows))
        else:
            print("Calibrator not enabled or not enough characteristic points")

        return transformed_frame, H

    @staticmethod
    def transform_frame(frame, H):
        rows, columns, channels = frame.shape
        transformed_frame = cv2.warpPerspective(frame, H, (columns, rows))
        return transformed_frame
