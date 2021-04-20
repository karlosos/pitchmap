from pitchmap.homography.camera_movement_analyser import CameraMovementAnalyser
from pitchmap.cache_loader.camera_movement import CameraMovementLoader

import numpy as np
from scipy.signal import find_peaks
from icecream import ic
import math
from pitchmap.homography import matrix_interp

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
        self.temp_homographies = {}

        self.frames_spacing = 30

    def calibrate(self):
        if self.flag == False:
            # Load camera movement
            self.camera_movement_analyser.get_characteristic_points()
            self.camera_angles = self.camera_movement_analyser.x_cum_sum
            self.manual_tracker.fl.set_current_frame_position(1)

            # Set characteristic frames
            arg_min_x = self.camera_movement_analyser.arg_x_min
            arg_max_x = self.camera_movement_analyser.arg_x_max

            self.characteristic_frames_numbers = self.find_characteristic_frames()
            self.characteristic_frames_iterator = iter(self.characteristic_frames_numbers)
            print("Characteristic frames", self.characteristic_frames_numbers)
            print("Length of characterstic frames", len(self.characteristic_frames_numbers))

            # Clean previous homography
            # self.manual_tracker.homographies = {}

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
            print(self.manual_tracker.homographies_angle)
            frame_idx = 1
            self.flag = False
            self.interpolate()
            self.manual_tracker.fl.set_current_frame_position(frame_idx-1)
            self.manual_tracker.load_next_frame()

    def find_characteristic_frames(self):
        camera_angles = self.camera_angles
        regular_spacing = np.arange(1, self.manual_tracker.fl.get_frames_count(), self.frames_spacing)
        characteristic_frames = regular_spacing.tolist()

        characteristic_frames = self.add_minmax_in_intervals(characteristic_frames)

        # add last frame and sort
        characteristic_frames.append(len(camera_angles))
        characteristic_frames = np.unique(characteristic_frames)

        # plot characteristic frames on camera_angles
        import matplotlib.pyplot as plt
        plt.plot(camera_angles)
        for characteristic_frame in characteristic_frames:
            plt.axvline(x=characteristic_frame-1, color='k', linestyle='--')
        plt.show()

        return characteristic_frames

    def add_minmax_in_intervals(self, characteristic_frames):
        """
        Add mins and max in every interval (from characteristic frame to next characteristic frame)
        """
        added_frame_in_loop = True
        while added_frame_in_loop:
            additional_frames = []
            length_before = len(characteristic_frames)
            for characteristic_frame in characteristic_frames:
                curr_range = self.camera_angles[characteristic_frame - 1:-1]
                if len(curr_range > 0):
                    additional_frames.append(characteristic_frame + np.argmax(curr_range))
                    additional_frames.append(characteristic_frame + np.argmin(curr_range))
            characteristic_frames = characteristic_frames + additional_frames
            characteristic_frames = np.unique(characteristic_frames).tolist()
            length_after = len(characteristic_frames)
            if length_after > length_before:
                print("Added frames in loop")
                print(characteristic_frames)
            else:
                print("Finished adding characteristic frames")
                added_frame_in_loop = False
        return characteristic_frames

    def interpolate(self):
        # Perform interpolation based on characteristic frames and homographies using
        # infomation about camera movement
        characteristic_frames = self.characteristic_frames_numbers
        characteristic_homographies = [self.manual_tracker.homographies_angle[i] for i in characteristic_frames]
        print("Interpolation for automatic homography steps")

        for i in range(len(characteristic_frames)):
            h1 = characteristic_homographies[i]
            f1 = characteristic_frames[i]
            print("")
            print(f"Loop i: {i}, f1: {f1}")
            for j in range(len(characteristic_frames)-1, i, -1):
                h2 = characteristic_homographies[j]
                f2 = characteristic_frames[j]
                print(f"Loop i: {i}, j: {j}, f2: {f2}")
                try:
                    self.interpolate_between_frames(f1, f2, h1, h2)
                except Exception as e:
                    print(f"Error while performing complex interpolation from {f1} to {f2}")
                    print(e)
        self.manual_tracker.homographies_angle = self.temp_homographies

    def interpolate_between_frames(self, f1, f2, h1, h2):
        camera_angles = self.camera_angles
        print(f"Trying interpolation from {f1} to {f2}")
        angle_1 = camera_angles[f1-1]
        angle_2 = camera_angles[f2-1]
        min_max_condition = self.check_min_max_condition(angle_1, angle_2, camera_angles, f1, f2)
        if min_max_condition:
            steps = np.abs(angle_2 - angle_1)
            # TODO: fix interpolation. For some cases interpolation makes rotation and not smooth transition. Why?
            part_homographies = matrix_interp.interpolate_transformation_matrices(0, math.ceil(steps) + 1, h1, h2)
            for j in range(f2 - f1):
                homography_index = np.abs(math.floor(camera_angles[f1 + j]) - math.floor(camera_angles[f1]))
                h = part_homographies[:, :, homography_index]
                self.temp_homographies[f1 + j] = h
            self.temp_homographies[f1] = h1
            self.temp_homographies[f2] = h2
            print(f"Performed complex interpolation from {f1} to {f2}")
        else:
            print(f"From {f1} to {f2} failed. Min/max check not passed.")

    def check_min_max_condition(self, angle_1, angle_2, camera_angles, f1, f2):
        max_condition = np.max(camera_angles[f1:f2]) <= np.max((angle_1, angle_2))
        min_condition = np.min(camera_angles[f1:f2]) >= np.min((angle_1, angle_2))
        print('max_condition', max_condition, angle_1, angle_2, f1, f2)
        print('min_condition', min_condition, angle_1, angle_2, f1, f2)
        return max_condition and min_condition
