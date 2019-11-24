from pitchmap.frame import mask
from pitchmap.homography import camera_movement_analyser

from abc import ABC, abstractmethod
import imutils
import numpy as np
import math


class CalibrationInteractor(ABC):
    @abstractmethod
    def __init__(self, pitch_map, calibrator, frame_loader):
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader
        self.__pitch_map = pitch_map

    @abstractmethod
    def start_calibration(self):
        pass

    @abstractmethod
    def perform_transform(self, players_list, team_colors):
        pass

    @abstractmethod
    def accept_transform(self):
        pass

    @abstractmethod
    def reset_transform(self):
        pass

    @abstractmethod
    def get_homography(self, frame_number):
        pass


class CalibrationInteractorAutomatic(CalibrationInteractor):
    def __init__(self, pitch_map, calibrator, frame_loader):
        super().__init__(pitch_map, calibrator, frame_loader)
        self.__pitch_map = pitch_map
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader

        self.__camera_movement_analyser = camera_movement_analyser.CameraMovementAnalyser(self.__frame_loader)
        self.__characteristic_frames_numbers = None
        self.__characteristic_frames_iterator = None
        arg_min_x, arg_max_x, min_x, max_x = self.__camera_movement_analyser.calculate_characteristic_points()
        self.__characteristic_frames_numbers = (arg_min_x, arg_max_x)
        self.__characteristic_frames_iterator = iter(self.__characteristic_frames_numbers)

        self.__current_calibration_frame_idx = None
        self.__current_homography = None
        self.__homographies_for_characteristic_frames = []
        self.homographies = np.array([])

        self.__started_calibrating_flag = False

    def start_calibration(self):
        self.__calibrator.clear_points()
        if self.__current_calibration_frame_idx is None:
            self.__calibrator.toggle_enabled()

        if not self.__started_calibrating_flag:
            try:
                self.__current_calibration_frame_idx = next(self.__characteristic_frames_iterator)
            except StopIteration:
                self.__current_calibration_frame_idx = -1
                return False

        if self.__current_calibration_frame_idx >= 0:
            self.__frame_loader.set_current_frame_position(self.__current_calibration_frame_idx)
            frame = self.__frame_loader.load_frame()
            frame = imutils.resize(frame, width=600)
            grass_mask = mask.grass(frame)
            self.__pitch_map.out_frame = grass_mask
            return True
        return False

    def perform_transform(self, players_list, team_colors):
        if self.__calibrator.can_perform_calibrate():
            players = players_list.get_players_positions_from_frame(
                frame_number=self.__frame_loader.get_current_frame_position())
            team_ids = players_list.get_players_team_ids_from_frame(
                frame_number=self.__frame_loader.get_current_frame_position())
            colors = list(map(lambda x: team_colors[x], team_ids))
            players_2d_positions, transformed_frame, H = self.__calibrator.calibrate(self.__pitch_map.out_frame, players,
                                                                                   colors)
            self.__pitch_map.out_frame = transformed_frame
            self.__pitch_map.add_players_to_model(players_2d_positions, colors)
            self.__current_homography = H

    def accept_transform(self):
        self.__homographies_for_characteristic_frames.append(self.__current_homography)
        if not self.start_calibration():
            self.interpolate()

    def reset_transform(self):
        # TODO:
        # 1. Reload frame
        # 2. Clear pitch model
        # 3. Clear points in calibration
        pass

    def interpolate(self):
        characteristic_homographies = self.__homographies_for_characteristic_frames

        steps = np.abs(self.__camera_movement_analyser.x_max - self.__camera_movement_analyser.x_min)
        self.homographies = self.__calibrator.interpolate(steps, characteristic_homographies[0],
                                                          characteristic_homographies[1])
        self.__pitch_map.set_transforming_flag(True)
        self.__calibrator.toggle_enabled()

    def get_homography(self, frame_number):
        camera_angle = self.__camera_movement_analyser.x_cumsum[frame_number - 1]
        min_x = self.__camera_movement_analyser.x_min
        print(
            f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
        h = self.homographies[:, :, math.floor(camera_angle - min_x)]
        return h

    def is_homography_exist(self, frame):
        s = self.homographies.shape
        try:
            _ = s[2]
            return True
        except IndexError:
            return False


class CalibrationInteractorSimple(CalibrationInteractor):
    def __init__(self, pitch_map, calibrator, frame_loader):
        super().__init__(pitch_map, calibrator, frame_loader)
        self.__pitch_map = pitch_map
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader

        self.homographies = {}
        self.__started_calibrating_flag = False

    def start_calibration(self):
        if not self.__calibrator.enabled:
            self.__pitch_map.create_model_window()

        if not self.__started_calibrating_flag:
            status = self.__calibrator.toggle_enabled()
            self.__pitch_map.load_frame()
            return status
        else:
            return False

    def perform_transform(self, players_list, team_colors):
        if self.__calibrator.can_perform_calibrate():
            if self.__calibrator.stop_calibration_H is None:
                players = players_list.get_players_positions_from_frame(
                    frame_number=self.__frame_loader.get_current_frame_position())
                team_ids = players_list.get_players_team_ids_from_frame(
                    frame_number=self.__frame_loader.get_current_frame_position())
                colors = list(map(lambda x: team_colors[x], team_ids))
                players_2d_positions, transformed_frame, H = self.__calibrator.calibrate(self.__pitch_map.out_frame,
                                                                                         players, colors)
                self.__pitch_map.out_frame = transformed_frame
                self.__pitch_map.add_players_to_model(players_2d_positions, colors)

                if self.__calibrator.start_calibration_H is None:
                    print("Start calibration")
                    self.__calibrator.start_calibration(H, self.__frame_loader.get_current_frame_position())
                elif self.__calibrator.stop_calibration_H is None:
                    print("Stop calibration")
                    self.__calibrator.end_calibration(H, self.__frame_loader.get_current_frame_position())
                    self.homographies = self.__calibrator.H_dictionary

    def accept_transform(self):
        if self.__calibrator.stop_calibration_H is not None:
            self.__calibrator.enabled = False
            self.__frame_loader.set_current_frame_position(self.__calibrator.start_calibration_frame_index)

    def reset_transform(self):
        pass

    def interpolate(self):
        pass

    def get_homography(self, frame_number):
        return self.homographies.get(frame_number, None)

    def is_homography_exist(self, frame):
        h = self.homographies.get(frame, None)
        if h is None:
            return False
        if type(h) is np.ndarray:
            return True
        else:
            return False
