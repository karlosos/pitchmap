import numpy as np
import cv2


class Calibrator:
    def __init__(self):
        self.enabled = False
        self.points = {}

        self.current_point = None

    def toggle_enabled(self):
        if not self.enabled:
            self.current_point = None

        self.enabled = not self.enabled

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
            print(self.points)
            self.current_point = None
            return index
        else:
            return False

    def get_points_count(self):
        return len(self.points)

    def calibrate(self, frame, players, players_colors):
        players_2d_positions = None
        transformed_frame = None

        if self.enabled and \
                self.get_points_count() >= 4:
            original_points, model_points = zip(*self.points.values())
            original_points = np.float32(original_points)
            model_points = np.float32(model_points)
            rows, columns, channels = frame.shape

            M, _ = cv2.findHomography(original_points, model_points)
            transformed_frame = cv2.warpPerspective(frame, M, (columns, rows))

            players = np.float32(players)
            players_2d_positions = []

            for player in players:
                player = np.array(player)
                player = np.append(player, 1.)
                # https://www.learnopencv.com/homography-examples-using-opencv-python-c/
                # calculating new positions
                player_2d_position = M.dot(player)
                player_2d_position = player_2d_position / player_2d_position[2]
                players_2d_positions.append(player_2d_position)

        return players_2d_positions, transformed_frame
