from pitchmap.homography.camera_movement_analyser import CameraMovementAnalyser
from pitchmap.cache_loader.camera_movement import CameraMovementLoader

import numpy as np
from scipy.signal import find_peaks
from icecream import ic


class AutomaticCalibrator:
    def __init__(self, manual_tracker):
        self.manual_tracker = manual_tracker
        self.camera_movement_analyser = CameraMovementAnalyser(self.manual_tracker.fl)
        camera_loader = CameraMovementLoader(self.camera_movement_analyser, self.manual_tracker.video_name)
        self.camera_movement_analyser.loader = camera_loader

        self.characteristic_frames_numbers = None
        self.characteristic_frames_iterator = None

        self.current_calibration_frame_idx = None
        self.camera_angles = None

        self.flag = False

    def calibrate(self):
        if self.flag == False:
            # Load camera movement
            self.camera_movement_analyser.get_characteristic_points()
            self.camera_angles = self.camera_movement_analyser.x_cum_sum
            self.manual_tracker.fl.set_current_frame_position(1)

            # Set characteristic frames
            arg_min_x = self.camera_movement_analyser.arg_x_min
            arg_max_x = self.camera_movement_analyser.arg_x_max

            self.characteristic_frames_numbers = self.find_characteristic_frames(self.camera_angles, spacing=150)
            self.characteristic_frames_iterator = iter(self.characteristic_frames_numbers)
            print("Characteristic frames", self.characteristic_frames_numbers)

            # Clean previous homography
            self.manual_tracker.homographies = {}

            # Perform transformation for all characteristic frames
            # and store corresponding homgoraphies
            self.flag = True
            self.manual_tracker.calibration()
            self.step()
        else:
            self.step()

    def step(self):
        try:
            print("Loading next characteristic frame")
            frame_idx = next(self.characteristic_frames_iterator)
            self.manual_tracker.fl.set_current_frame_position(frame_idx-1)
            self.manual_tracker.load_next_frame()
        except StopIteration:
            print("Finished automatic homography steps")
            print(self.manual_tracker.homographies)
            frame_idx = 1
            self.flag = False
            self.interpolate()
            self.manual_tracker.fl.set_current_frame_position(frame_idx-1)
            self.manual_tracker.load_next_frame()

    def find_characteristic_frames(self, camera_angles, spacing):
        regular_spacing = np.arange(1, self.manual_tracker.fl.get_frames_count(), spacing)
        characteristic_frames = regular_spacing.tolist()
        additional_frames = []

        for characteristic_frame in characteristic_frames:
            curr_range = self.camera_angles[characteristic_frame-1:-1]
            additional_frames.append(characteristic_frame + np.argmax(curr_range))
            additional_frames.append(characteristic_frame + np.argmin(curr_range))

        # add last frame and sort
        characteristic_frames.append(len(camera_angles))
        characteristic_frames = np.unique(np.concatenate((characteristic_frames, additional_frames)))

        # plot characteristic frames on camera_angles
        import matplotlib.pyplot as plt
        plt.plot(camera_angles)
        for characteristic_frame in characteristic_frames:
            plt.axvline(x=characteristic_frame-1, color='k', linestyle='--')
        plt.show()

        return characteristic_frames

    def interpolate(self):
        # Perform interpolation based on characteristic frames and homographies using
        # infomation about camera movement
        print("Interpolation for automatic homography steps")
        pass

    # def interpolate(self):
    #     characteristic_homographies = self.__homographies_for_characteristic_frames
    #     camera_angles = self.__camera_movement_analyser.x_cum_sum
    #     print(self.__characteristic_frames_numbers, len(self.__characteristic_frames_numbers))
    #     print(characteristic_homographies, len(characteristic_homographies))
    #
    #     # First step: homography interpolation for every angle from min to max
    #     steps = np.abs(self.__camera_movement_analyser.x_max - self.__camera_movement_analyser.x_min)
    #     homography_min = characteristic_homographies[-2]
    #     homography_max = characteristic_homographies[-1]
    #     self.homographies_angle = self.__calibrator.interpolate(steps, homography_min,
    #                                                             homography_max)
    #
    #     # TODO: change to frame numers from loader
    #     self.homographies = [None for i in range(self.__characteristic_frames_numbers[-3]+3)]
    #
    #     # Advanced interpolation loop
    #     peak_frames = self.__characteristic_frames_numbers[:-2]
    #     for i in range(len(peak_frames)):
    #         h1 = characteristic_homographies[i]
    #         f1 = peak_frames[i]
    #         print(f"Loop i: {i}, f1: {f1}")
    #         if h1 is None:
    #             h1 = self.get_homography(f1+1)
    #         for j in range(len(peak_frames)-1, i, -1):
    #             h2 = characteristic_homographies[j]
    #             f2 = peak_frames[j]
    #             print(f"Loop i: {i}, j: {j}, f2: {f2}")
    #             if h2 is None:
    #                 h2 = self.get_homography(f2+1)
    #             try:
    #                 self.interpolate_between_frames(camera_angles, f1, f2, h1, h2)
    #             except Exception as e:
    #                 print(f"Error while performing complex interpolation from {f1} to {f2}")
    #                 print(e)
    #
    #     self.__pitch_map.set_transforming_flag(True)
    #     self.__calibrator.toggle_enabled()
    #
    # def interpolate_between_frames(self, camera_angles, f1, f2, h1, h2, recursion=False):
    #     print(f"Trying interpolation from {f1} to {f2}")
    #     angle_1 = camera_angles[f1]
    #     angle_2 = camera_angles[f2]
    #     if np.max(camera_angles[f1:f2]) <= np.max((angle_1, angle_2)) and np.min(camera_angles[f1:f2]) >= np.min(
    #             (angle_1, angle_2)):
    #         steps = np.abs(angle_2 - angle_1)
    #         part_homographies = self.__calibrator.interpolate(steps, h1, h2)
    #         for j in range(f2 - f1):
    #             homography_index = np.abs(math.floor(camera_angles[f1 + j]) - math.floor(camera_angles[f1]))
    #             h = part_homographies[:, :, homography_index]
    #             self.homographies[f1 + j] = h
    #         self.homographies[f1] = h1
    #         self.homographies[f2] = h2
    #         print(f"Performed complex interpolation from {f1} to {f2}")
    #     else:
    #         print(f"From {f1} to {f2} failed. Min/max check not passed.")
    #
    # def get_homography(self, frame_number):
    #     if self.homographies[frame_number] is not None:
    #         print(f"frame: {frame_number}: based on advanced interpolation")
    #         return self.homographies[frame_number]
    #     camera_angle = self.__camera_movement_analyser.x_cum_sum[frame_number - 1]
    #     min_x = self.__camera_movement_analyser.x_min
    #     # print(
    #     #     f"frame: {frame_number} camera_angle: {camera_angle}, obliczony index:{camera_angle - min_x}, {math.floor(camera_angle - min_x)}, min_x = {min_x}")
    #     h = self.homographies_angle[:, :, math.floor(camera_angle - min_x)]
    #     print(f"frame: {frame_number}: based on angle interpolation")
    #     return h
