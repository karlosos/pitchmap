from pitchmap.segmentation import mask
from pitchmap.homography import camera_movement_analyser
from pitchmap.cache_loader import camera_movement
from pitchmap.homography.keypoints_utils import _points_from_mask
from pitchmap.homography.homography_models import KeypointDetectorModel

from abc import ABC, abstractmethod
import imutils
import numpy as np
import math
import cv2
from scipy.signal import find_peaks


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
        camera_loader = camera_movement.CameraMovementLoader(self.__camera_movement_analyser,
                                                             self.__pitch_map.video_name)
        self.__camera_movement_analyser.loader = camera_loader

        self.__characteristic_frames_numbers = None
        self.__characteristic_frames_iterator = None
        arg_min_x, arg_max_x, min_x, max_x = self.__camera_movement_analyser.get_characteristic_points()
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
        camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
        min_x = self.__camera_movement_analyser.x_min
        print(
            f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
        h = self.homographies[:, :, math.floor(camera_angle - min_x)]
        return h

    def is_homography_exist(self, frame_number):
        s = self.homographies.shape
        try:
            _ = s[2]
            x = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
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


class CalibrationInteractorMiddlePoint(CalibrationInteractor):
    def __init__(self, pitch_map, calibrator, frame_loader):
        super().__init__(pitch_map, calibrator, frame_loader)
        self.__pitch_map = pitch_map
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader

        self.__camera_movement_analyser = camera_movement_analyser.CameraMovementAnalyser(self.__frame_loader)
        camera_loader = camera_movement.CameraMovementLoader(self.__camera_movement_analyser,
                                                             self.__pitch_map.video_name)
        self.__camera_movement_analyser.loader = camera_loader

        self.__characteristic_frames_numbers = None
        self.__characteristic_frames_iterator = None
        arg_min_x, arg_max_x, min_x, max_x = self.__camera_movement_analyser.get_characteristic_points()
        mean_x = max_x - (max_x - min_x)/2
        mean_x, arg_mean_x = self.find_nearest(self.__camera_movement_analyser.x_cum_sum, mean_x)
        self.__mean_x = mean_x

        self.__characteristic_frames_numbers = (arg_min_x, arg_mean_x, arg_max_x)
        self.__characteristic_frames_iterator = iter(self.__characteristic_frames_numbers)

        self.__current_calibration_frame_idx = None
        self.__current_homography = None
        self.__homographies_for_characteristic_frames = []
        self.homographies = np.array([])

        self.__started_calibrating_flag = False

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

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

        steps_left_middle = np.abs(self.__camera_movement_analyser.x_max - self.__mean_x)
        homographies_left_middle = self.__calibrator.interpolate(steps_left_middle, characteristic_homographies[0],
                                                                 characteristic_homographies[1])

        steps_middle_right = np.abs(self.__mean_x - self.__camera_movement_analyser.x_min)
        homographies_middle_right = self.__calibrator.interpolate(steps_middle_right, characteristic_homographies[1],
                                                                  characteristic_homographies[2])

        self.homographies = np.concatenate((homographies_left_middle, homographies_middle_right), axis=2)

        self.__pitch_map.set_transforming_flag(True)
        self.__calibrator.toggle_enabled()

    def get_homography(self, frame_number):
        try:
            camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
        except IndexError:
            camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 2]

        min_x = self.__camera_movement_analyser.x_min
        # print(
        #     f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
        h = self.homographies[:, :, math.floor(camera_angle - min_x)]
        return h

    def is_homography_exist(self, frame):
        s = self.homographies.shape
        try:
            _ = s[2]
            return True
        except IndexError:
            return False


class CalibrationInteractorKeypoints(CalibrationInteractor):
    def __init__(self, pitch_map, calibrator, frame_loader):
        super().__init__(pitch_map, calibrator, frame_loader)
        self.__pitch_map = pitch_map
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader

        self.__camera_movement_analyser = camera_movement_analyser.CameraMovementAnalyser(self.__frame_loader)
        camera_loader = camera_movement.CameraMovementLoader(self.__camera_movement_analyser,
                                                             self.__pitch_map.video_name)
        self.__camera_movement_analyser.loader = camera_loader

        self.__characteristic_frames_numbers = None
        self.__characteristic_frames_iterator = None
        arg_min_x, arg_max_x, min_x, max_x = self.__camera_movement_analyser.get_characteristic_points()
        # TODO: find more characteristic frame number
        self.__characteristic_frames_numbers = (arg_min_x, arg_max_x)
        self.__characteristic_frames_iterator = iter(self.__characteristic_frames_numbers)

        self.__current_calibration_frame_idx = None
        self.__current_homography = None
        self.__homographies_for_characteristic_frames = []
        self.homographies = np.array([])

        self.__started_calibrating_flag = False

        self.__kp_model = KeypointDetectorModel(
            backbone="efficientnetb3",
            num_classes=29,
            input_shape=(320, 320),
        )

        self.__kp_model.load_weights("./models/FPN_efficientnetb3_0.0001_8_427.h5")


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

            # Finding keypoints with KeypointsModel
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pr_mask = self.__kp_model(image)
            src_points, dst_points = _points_from_mask(pr_mask[0])

            # Rescaling because model works on 320x320 input
            height, width, _ = image.shape
            height_ratio = 320 / height
            width_ratio = 320 / width

            for idx, src_point in enumerate(src_points):
                dst_point = dst_points[idx]
                src_point = [src_point[0]/width_ratio, src_point[1]/height_ratio]
                self.__pitch_map.display.add_point_main_window(int(src_point[0]), int(src_point[1]))
                self.__pitch_map.display.add_point_model_window(int(dst_point[0]), int(dst_point[1]))
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
        # TODO: edit interpolation for more calibration frames
        characteristic_homographies = self.__homographies_for_characteristic_frames

        steps = np.abs(self.__camera_movement_analyser.x_max - self.__camera_movement_analyser.x_min)
        self.homographies = self.__calibrator.interpolate(steps, characteristic_homographies[0],
                                                          characteristic_homographies[1])
        self.__pitch_map.set_transforming_flag(True)
        self.__calibrator.toggle_enabled()

    def get_homography(self, frame_number):
        camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
        min_x = self.__camera_movement_analyser.x_min
        print(
            f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
        h = self.homographies[:, :, math.floor(camera_angle - min_x)]
        return h

    def is_homography_exist(self, frame_number):
        s = self.homographies.shape
        try:
            _ = s[2]
            x = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
            return True
        except IndexError:
            return False


class CalibrationInteractorKeypointsComplex(CalibrationInteractor):
    def __init__(self, pitch_map, calibrator, frame_loader):
        super().__init__(pitch_map, calibrator, frame_loader)
        self.__pitch_map = pitch_map
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader

        self.__camera_movement_analyser = camera_movement_analyser.CameraMovementAnalyser(self.__frame_loader)
        camera_loader = camera_movement.CameraMovementLoader(self.__camera_movement_analyser,
                                                             self.__pitch_map.video_name)
        self.__camera_movement_analyser.loader = camera_loader

        self.__characteristic_frames_numbers = None
        self.__characteristic_frames_iterator = None
        arg_min_x, arg_max_x, min_x, max_x = self.__camera_movement_analyser.get_characteristic_points()  # this only load data for camera analyser
        camera_angles = self.__camera_movement_analyser.x_cum_sum
        characteristic_frames = self.find_characteristic_frames(camera_angles)
        self.__characteristic_frames_numbers = np.concatenate((characteristic_frames, (arg_min_x, arg_max_x)))
        self.__characteristic_frames_iterator = iter(self.__characteristic_frames_numbers)

        self.__current_calibration_frame_idx = None
        self.__current_homography = None
        self.__homographies_for_characteristic_frames = []
        self.homographies_angle = np.array([])

        self.__started_calibrating_flag = False

        self.__kp_model = KeypointDetectorModel(
            backbone="efficientnetb3",
            num_classes=29,
            input_shape=(320, 320),
        )

        self.__kp_model.load_weights("./models/FPN_efficientnetb3_0.0001_8_427.h5")

    def find_characteristic_frames(self, camera_angles):
        min_peaks, _ = find_peaks(-camera_angles, width=3)
        max_peaks, _ = find_peaks(camera_angles, width=3)
        # TODO: add min and max for normal interpolation
        characteristic_frames = np.concatenate(([0], min_peaks, max_peaks, [len(camera_angles) - 1]))
        return np.sort(characteristic_frames)

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

            # Finding keypoints with KeypointsModel
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pr_mask = self.__kp_model(image)
            src_points, dst_points = _points_from_mask(pr_mask[0])
            # TODO: jeżeli mniej niż 4 punkty to przerwij

            # Rescaling because model works on 320x320 input
            height, width, _ = image.shape
            height_ratio = 320 / height
            width_ratio = 320 / width

            for idx, src_point in enumerate(src_points):
                dst_point = dst_points[idx]
                src_point = [src_point[0]/width_ratio, src_point[1]/height_ratio]
                self.__pitch_map.display.add_point_main_window(int(src_point[0]), int(src_point[1]))
                self.__pitch_map.display.add_point_model_window(int(dst_point[0]), int(dst_point[1]))
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
        else:
            self.__current_homography = None

    def accept_transform(self):
        self.__homographies_for_characteristic_frames.append(self.__current_homography)
        if not self.start_calibration():
            self.interpolate()

    def cancel_transform(self):
        self.__homographies_for_characteristic_frames.append(None)  # self.__current_homography should be None too,
        # but for a "safety" reason I wanted to be explicit
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
        camera_angles = self.__camera_movement_analyser.x_cum_sum
        print(self.__characteristic_frames_numbers, len(self.__characteristic_frames_numbers))
        print(characteristic_homographies, len(characteristic_homographies))

        # TODO: edit interpolation for more calibration frames
        steps = np.abs(self.__camera_movement_analyser.x_max - self.__camera_movement_analyser.x_min)
        homography_min = characteristic_homographies[-2]
        homography_max = characteristic_homographies[-1]
        self.homographies_angle = self.__calibrator.interpolate(steps, homography_min,
                                                                homography_max)
        self.homographies = [None for i in range(self.__characteristic_frames_numbers[-3]+1)]

        # Complex interpolation loop
        for i in range(len(self.__characteristic_frames_numbers)-3):  # for every characteristic without last two (min, max)
            h1 = characteristic_homographies[i]
            h2 = characteristic_homographies[i+1]
            f1 = self.__characteristic_frames_numbers[i]
            f2 = self.__characteristic_frames_numbers[i+1]

            if h1 is not None and h2 is not None:
                try:
                    self.interpolate_between_frames(camera_angles, f1, f2, h1, h2)
                except Exception as e:
                    print(f"Error while performing complex interpolation from {f1} to {f2}")
                    print(e)
            else:
                print(f"h1/h2 is None for complex interpolation {f1} to {f2}")
                if h1 is None:
                    h1 = self.get_homography(f1+1)
                if h2 is None:
                    h2 = self.get_homography(f2+1)
                try:
                    self.interpolate_between_frames(camera_angles, f1, f2, h1, h2)
                except Exception as e:
                    print(f"Error while performing complex interpolation from {f1} to {f2}")
                    print(e)

        self.__pitch_map.set_transforming_flag(True)
        self.__calibrator.toggle_enabled()

    def interpolate_between_frames(self, camera_angles, f1, f2, h1, h2, recursion=False):
        angle_1 = camera_angles[f1]
        angle_2 = camera_angles[f2]
        if np.max(camera_angles[f1:f2]) <= np.max((angle_1, angle_2)) and np.min(camera_angles[f1:f2]) >= np.min(
                (angle_1, angle_2)):
            steps = np.abs(angle_2 - angle_1)
            part_homographies = self.__calibrator.interpolate(steps, h1, h2)
            for j in range(f2 - f1):
                homography_index = np.abs(math.floor(camera_angles[f1 + j]) - math.floor(camera_angles[f1]))
                h = part_homographies[:, :, homography_index]
                self.homographies[f1 + j] = h
            self.homographies[f1] = h1
            self.homographies[f2] = h2
            print(f"Performed complex interpolation from {f1} to {f2}")
        else:
            print(f"Failed to perform complex interpolation from {f1} to {f2}")
            print(f"Max/min angle is not at first/last frame")
            if not recursion:
                print("Fixing ")
                if np.max(camera_angles[f1:f2]) > np.max((angle_1, angle_2)):
                    medium_pos = np.argmax(camera_angles[f1:f2])
                    fm = f1+medium_pos
                    hm = self.get_homography(fm)
                    self.interpolate_between_frames(camera_angles, f1, fm, h1, hm, recursion=True)
                    self.interpolate_between_frames(camera_angles, fm, f2, hm, h2, recursion=True)
                else:
                    medium_pos = np.argmin(camera_angles[f1:f2])
                    fm = f1+medium_pos
                    hm = self.get_homography(fm)
                    self.interpolate_between_frames(camera_angles, f1, fm, h1, hm, recursion=True)
                    self.interpolate_between_frames(camera_angles, fm, f2, hm, h2, recursion=True)
                print("Fixed")

    def get_homography(self, frame_number):
        if self.homographies[frame_number] is not None:
            print(f"frame: {frame_number}: based on complex interpolation")
            return self.homographies[frame_number]
        camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
        min_x = self.__camera_movement_analyser.x_min
        # print(
        #     f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
        h = self.homographies_angle[:, :, math.floor(camera_angle - min_x)]
        return h

    def is_homography_exist(self, frame_number):
        s = self.homographies_angle.shape
        try:
            _ = s[2]
            x = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
            return True
        except IndexError:
            return False

class CalibrationInteractorKeypointsAdvanced(CalibrationInteractor):
    def __init__(self, pitch_map, calibrator, frame_loader):
        super().__init__(pitch_map, calibrator, frame_loader)
        self.__pitch_map = pitch_map
        self.__calibrator = calibrator
        self.__frame_loader = frame_loader

        self.__camera_movement_analyser = camera_movement_analyser.CameraMovementAnalyser(self.__frame_loader)
        camera_loader = camera_movement.CameraMovementLoader(self.__camera_movement_analyser,
                                                             self.__pitch_map.video_name)
        self.__camera_movement_analyser.loader = camera_loader

        self.__characteristic_frames_numbers = None
        self.__characteristic_frames_iterator = None
        arg_min_x, arg_max_x, min_x, max_x = self.__camera_movement_analyser.get_characteristic_points()  # this only load data for camera analyser
        camera_angles = self.__camera_movement_analyser.x_cum_sum
        characteristic_frames = self.find_characteristic_frames(camera_angles)
        self.__characteristic_frames_numbers = np.concatenate((characteristic_frames, (arg_min_x, arg_max_x)))
        self.__characteristic_frames_iterator = iter(self.__characteristic_frames_numbers)

        self.__current_calibration_frame_idx = None
        self.__current_homography = None
        self.__homographies_for_characteristic_frames = []
        self.homographies_angle = np.array([])

        self.__started_calibrating_flag = False

        self.__kp_model = KeypointDetectorModel(
            backbone="efficientnetb3",
            num_classes=29,
            input_shape=(320, 320),
        )

        self.__kp_model.load_weights("./models/FPN_efficientnetb3_0.0001_8_427.h5")

    def find_characteristic_frames(self, camera_angles):
        min_peaks, _ = find_peaks(-camera_angles, width=3)
        max_peaks, _ = find_peaks(camera_angles, width=3)
        # TODO: add min and max for normal interpolation
        characteristic_frames = np.concatenate(([0], min_peaks, max_peaks, [len(camera_angles) - 1]))
        return np.sort(characteristic_frames)

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

            # Finding keypoints with KeypointsModel
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pr_mask = self.__kp_model(image)
            src_points, dst_points = _points_from_mask(pr_mask[0])
            # TODO: jeżeli mniej niż 4 punkty to przerwij EDIT: to jest robione w perform transform

            # Rescaling because model works on 320x320 input
            height, width, _ = image.shape
            height_ratio = 320 / height
            width_ratio = 320 / width

            for idx, src_point in enumerate(src_points):
                dst_point = dst_points[idx]
                src_point = [src_point[0]/width_ratio, src_point[1]/height_ratio]
                self.__pitch_map.display.add_point_main_window(int(src_point[0]), int(src_point[1]))
                self.__pitch_map.display.add_point_model_window(int(dst_point[0]), int(dst_point[1]))
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
        else:
            self.__current_homography = None

    def accept_transform(self):
        self.__homographies_for_characteristic_frames.append(self.__current_homography)
        if not self.start_calibration():
            self.interpolate()

    def cancel_transform(self):
        self.__homographies_for_characteristic_frames.append(None)  # self.__current_homography should be None too,
        # but for a "safety" reason I wanted to be explicit
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
        camera_angles = self.__camera_movement_analyser.x_cum_sum
        print(self.__characteristic_frames_numbers, len(self.__characteristic_frames_numbers))
        print(characteristic_homographies, len(characteristic_homographies))

        # First step: homography interpolation for every angle from min to max
        steps = np.abs(self.__camera_movement_analyser.x_max - self.__camera_movement_analyser.x_min)
        homography_min = characteristic_homographies[-2]
        homography_max = characteristic_homographies[-1]
        self.homographies_angle = self.__calibrator.interpolate(steps, homography_min,
                                                                homography_max)

        self.homographies = [None for i in range(self.__characteristic_frames_numbers[-3]+1)]

        # Advanced interpolation loop
        peak_frames = self.__characteristic_frames_numbers[:-2]
        for i in range(len(peak_frames)):
            h1 = characteristic_homographies[i]
            f1 = peak_frames[i]
            print(f"Loop i: {i}, f1: {f1}")
            if h1 is None:
                h1 = self.get_homography(f1+1)
            for j in range(len(peak_frames)-1, i, -1):
                h2 = characteristic_homographies[j]
                f2 = peak_frames[j]
                print(f"Loop i: {i}, j: {j}, f2: {f2}")
                if h2 is None:
                    h2 = self.get_homography(f2+1)
                try:
                    self.interpolate_between_frames(camera_angles, f1, f2, h1, h2)
                except Exception as e:
                    print(f"Error while performing complex interpolation from {f1} to {f2}")
                    print(e)

        self.__pitch_map.set_transforming_flag(True)
        self.__calibrator.toggle_enabled()

    def interpolate_between_frames(self, camera_angles, f1, f2, h1, h2, recursion=False):
        print(f"Trying interpolation from {f1} to {f2}")
        angle_1 = camera_angles[f1]
        angle_2 = camera_angles[f2]
        if np.max(camera_angles[f1:f2]) <= np.max((angle_1, angle_2)) and np.min(camera_angles[f1:f2]) >= np.min(
                (angle_1, angle_2)):
            steps = np.abs(angle_2 - angle_1)
            part_homographies = self.__calibrator.interpolate(steps, h1, h2)
            for j in range(f2 - f1):
                homography_index = np.abs(math.floor(camera_angles[f1 + j]) - math.floor(camera_angles[f1]))
                h = part_homographies[:, :, homography_index]
                self.homographies[f1 + j] = h
            self.homographies[f1] = h1
            self.homographies[f2] = h2
            print(f"Performed complex interpolation from {f1} to {f2}")
        else:
            print(f"From {f1} to {f2} failed. Min/max check not passed.")

    def get_homography(self, frame_number):
        if self.homographies[frame_number] is not None:
            print(f"frame: {frame_number}: based on advanced interpolation")
            return self.homographies[frame_number]
        camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
        min_x = self.__camera_movement_analyser.x_min
        # print(
        #     f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
        h = self.homographies_angle[:, :, math.floor(camera_angle - min_x)]
        print(f"frame: {frame_number}: based on angle interpolation")
        return h

    def is_homography_exist(self, frame_number):
        s = self.homographies_angle.shape
        try:
            _ = s[2]
            x = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
            return True
        except IndexError:
            return False
