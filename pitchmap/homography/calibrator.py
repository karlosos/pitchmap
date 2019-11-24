import numpy as np
import cv2
from pitchmap.homography import matrix_interp
import math


class Calibrator:
    def __init__(self):
        self.enabled = False
        self.points = {}

        self.current_point = None

        self.start_calibration_H = None
        self.stop_calibration_H = None

        self.start_calibration_frame_index = None
        self.stop_calibration_frame_index = None

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
            #print(self.points)
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

    def calibrate(self, frame, players, players_colors):
        players_2d_positions = None
        transformed_frame = None

        if self.enabled and \
                self.get_points_count() >= 4:
            original_points, model_points = zip(*self.points.values())
            original_points = np.float32(original_points)
            model_points = np.float32(model_points)
            rows, columns, channels = frame.shape

            H, _ = cv2.findHomography(original_points, model_points)
            transformed_frame = cv2.warpPerspective(frame, H, (columns, rows))

            players = np.float32(players)
            players_2d_positions = []

            for player in players:
                player = np.array(player)
                player = np.append(player, 1.)
                # https://www.learnopencv.com/homography-examples-using-opencv-python-c/
                # calculating new positions
                player_2d_position = H.dot(player)
                player_2d_position = player_2d_position / player_2d_position[2]
                players_2d_positions.append(player_2d_position)

        return players_2d_positions, transformed_frame, H

    @staticmethod
    def transform_to_2d(players, H):
        players = np.float32(players)
        players_2d_positions = []

        for player in players:
            player = np.array(player)
            player = np.append(player, 1.)
            # https://www.learnopencv.com/homography-examples-using-opencv-python-c/
            # calculating new positions
            player_2d_position = H.dot(player)
            player_2d_position = player_2d_position / player_2d_position[2]
            players_2d_positions.append(player_2d_position)

        return players_2d_positions

    def start_calibration(self, H, frame_index):
        self.start_calibration_H = H
        self.start_calibration_frame_index = frame_index

    def end_calibration(self, H_m, m):
        if m > self.start_calibration_frame_index:
            self.stop_calibration_frame_index = m
            self.stop_calibration_H = H_m
            H = matrix_interp.interpolate_transformation_matrices(self.start_calibration_frame_index,
                                                                  self.stop_calibration_frame_index,
                                                                  self.start_calibration_H, self.stop_calibration_H)
            H_dictionary = {}
            for k in range(int(self.stop_calibration_frame_index - self.start_calibration_frame_index)):
                print(H[:, :, k])
                H_dictionary[int(self.start_calibration_frame_index + k)] = H[:, :, k]
            self.H_dictionary.update(H_dictionary)
            return True
        else:
            return False

    def interpolate(self, steps, start_H, stop_H):
        H = matrix_interp.interpolate_transformation_matrices(0, math.ceil(steps) + 1, start_H, stop_H)
        return H

    def clear_interpolation(self):
        self.start_calibration_H = None
        self.stop_calibration_H = None

        self.start_calibration_frame_index = None
        self.stop_calibration_frame_index = None
