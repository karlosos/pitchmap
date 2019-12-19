import numpy as np
import cv2
from pitchmap.homography import matrix_interp
import math


class Calibrator:
    def __init__(self):
        self.enabled = False
        self.points = {}
        self.index_count = 1
        self.current_point = None

    def toggle_enabled(self):
        if not self.enabled:
            self.current_point = None
            # self.points = {}

        self.enabled = not self.enabled
        return self.enabled

    def clear_points(self):
        self.points = {}
        self.index_count = 1
        self.current_point = None

    def add_point_main_window(self, pos):
        if self.current_point is None:
            self.current_point = pos
            index = self.index_count
            self.points[index] = [pos, None]
            return index, self.points[index]
        else:
            return False, False

    def add_point_model_window(self, pos):
        if self.current_point is not None:
            index = self.index_count
            self.index_count += 1
            self.points[index][1] = pos
            self.current_point = None
            return index, self.points[index]
        else:
            return False, False

    def get_points_count(self):
        return len(self.points)

    def can_perform_calibrate(self):
        if self.enabled and self.get_points_count() >= 4:
            return True
        return False

    def find_homography(self, frame, pitch_model):
        transformed_frame = None

        if self.enabled and \
                self.get_points_count() >= 4:
            original_points, model_points = zip(*self.points.values())
            original_points = np.float32(original_points)
            model_points = np.float32(model_points)
            rows, columns, channels = pitch_model.shape

            H, _ = cv2.findHomography(original_points, model_points)
            transformed_frame = cv2.warpPerspective(frame, H, (columns, rows))
        else:
            print("Calibrator not enabled or not enough characteristic points")

        return transformed_frame, H

    @staticmethod
    def transform_frame(frame, H, pitch_model):
        rows, columns, channels = pitch_model.shape
        transformed_frame = cv2.warpPerspective(frame, H, (columns, rows))
        return transformed_frame

    def clean_calibration_points(self):
        print(self.points)
        keys_to_delete = []
        for key, value in self.points.items():
            if value[0] is None or value[1] is None:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.points[key]

        print(self.points)
        self.current_point = None
