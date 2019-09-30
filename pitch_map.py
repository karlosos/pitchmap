"""
Main file of PitchMap. All process trough loading images from video to displaying 2D map.
"""
import frame_loader
import frame_display
import mask
import detect
import calibrator
import team_detection
import tracker
import keyboard_actions

import imutils
import cv2
import numpy as np


class PitchMap:
    def __init__(self, tracking_method=None):
        """
        :param tracking_method: if left default (None) then there's no tracking and detection
        is performed in every frame
        """
        self.__video_name = 'Dynamic_Barca_Real.mp4'
        self.__window_name = f'PitchMap: {self.__video_name}'

        self.fl = frame_loader.FrameLoader(self.__video_name)
        self.calibrator = calibrator.Calibrator()
        self.__display = frame_display.Display(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               pitchmap=self, frame_count=self.fl.get_frames_count())

        self.players = []
        self.players_colors = []

        self.out_frame = None

        # Team detection initialization
        selected_frames_for_clustering = self.fl.select_frames_for_clustering()
        self.__team_detector = team_detection.TeamDetection()
        self.__team_detector.cluster_teams(selected_frames_for_clustering)

        # Players tracking initialization
        self.__tracker = tracker.Tracker(tracking_method)

        self.M_start = None
        self.M_stop = None

        self.frame_n = None
        self.frame_m = None

        self.displaying = False
        self.H_dictionary = None

    def loop(self):
        while True:
            if not self.calibrator.enabled:
                frame = self.fl.load_frame()
                frame = imutils.resize(frame, width=600)

                grass_mask = mask.grass(frame)
                edges = detect.edges_detection(grass_mask)
                lines_frame = detect.lines_detection(edges, grass_mask)

                bounding_boxes_frame, bounding_boxes, labels = self.__tracker.update(grass_mask)

                self.players = []
                self.players_colors = []
                self.draw_bounding_boxes(frame, grass_mask, bounding_boxes)

                self.out_frame = cv2.addWeighted(grass_mask, 0.8, lines_frame, 1, 0)

                if self.displaying:
                    frame_idx = self.fl.get_current_frame_position()
                    print(f"{frame_idx} < {self.frame_m}")
                    if frame_idx < self.frame_m:

                        players_2d_positions = self.calibrator.transform_to_2d(self.players, self.players_colors,
                                                                               self.H_dictionary[int(frame_idx)])
                        self.__display.show_model()
                        self.__display.add_players_to_model(players_2d_positions, self.players_colors)

            else:
                self.__display.show_model()

            self.__display.show(self.out_frame, self.fl.get_current_frame_position())

            key = cv2.waitKey(1) & 0xff
            is_exit = not keyboard_actions.key_pressed(key, self)
            if is_exit:
                break

        self.__display.close_windows()
        self.fl.release()

    def draw_bounding_boxes(self, frame, grass_mask, bounding_boxes):
        team_colors = [(35, 117, 250), (250, 46, 35), (255, 48, 241)]
        for idx, box in enumerate(bounding_boxes):
            player_color, (x, y) = self.__team_detector.color_detection_for_player(frame, box)
            team_id = self.__team_detector.team_detection_for_player(np.asarray(player_color))[0]
            team_color = team_colors[team_id]

            # TODO get team color based on team_id
            cv2.circle(grass_mask, (x, y), 3, team_color, 5)
            cv2.putText(grass_mask, text=str(team_id), org=(x + 3, y + 3),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), lineType=1)
            self.players.append((x, y))
            self.players_colors.append(team_color)

    def start_calibration(self):
        if not self.calibrator.enabled:
            self.__display.create_model_window()
            self.__display.clear_model()
            self.__display.show_model()
        else:
            self.__display.close_model_window()
        self.calibrator.toggle_enabled()

    def perform_transform(self):
        players_2d_positions, transformed_frame, M = self.calibrator.calibrate(self.out_frame, self.players,
                                                                                 self.players_colors)
        self.out_frame = transformed_frame
        self.__display.add_players_to_model(players_2d_positions, self.players_colors)

        if self.M_start is None:
            self.M_start = M
            self.frame_n = self.fl.get_current_frame_position()
            print("Saved M start")
        elif self.M_stop is None:
            m = self.fl.get_current_frame_position()
            if m > self.frame_n:
                self.frame_m = m
                self.M_stop = M
                H = self.calculate_transformation_matrices(self.frame_n, self.frame_m, self.M_start, self.M_stop)
                print(f"n: {self.frame_n}")
                print(f"m: {self.frame_m}")
                print(f"H_i: {self.M_start}")
                print(f"H_j: {self.M_stop}")

                H_dictionary = {}
                for k in range(int(self.frame_m - self.frame_n)):
                    print(H[:, :, k])
                    H_dictionary[int(self.frame_n + k)] = H[:, :, k]
                self.H_dictionary = H_dictionary
                print("Stored matrices in dictionary")
                # change current position to n
                self.fl.set_current_frame_position(self.frame_n)
                # change mode to prediction and showing 2d model
                self.displaying = True
                self.calibrator.enabled = False
            else:
                print(f"Frame must be greater than starting. Start {self.M_start}, stop: {m}")

    def testing_interpolation(self):
        H_i = np.array([[ 1.57240776e+00,  3.90147611e+00, -2.88507021e+01], [ 3.43698327e-01,  6.70651630e+00, -5.16118081e+02], [ 1.35467928e-03,  1.31340979e-02,  1.00000000e+00]])

        H_j = np.array([[ 1.11658938e+00,  2.80626456e+00, -7.49405476e+01],  [ 3.25307780e-02,  5.27465907e+00, -3.28946531e+02],  [ 5.75744925e-04,  9.32661876e-03,  1.00000000e+00]])

        n = 5
        m = 48

        H = self.calculate_transformation_matrices(n, m, H_i, H_j)

        H_dictionary = {}
        for k in range(int(m - n)):
            print(H[:, :, k])
            H_dictionary[int(n + k)] = H[:, :, k]
        self.H_dictionary = H_dictionary
        print("Stored matrices in dictionary")
        # change current position to n
        self.fl.set_current_frame_position(n)
        # change mode to prediction and showing 2d model
        self.displaying = True
        self.calibrator.enabled = False

        self.frame_m = m
        self.frame_n = n
        self.M_start = H_i
        self.M_stop = H_j

    def calculate_transformation_matrices(self, n, m, H_n, H_m):
        H = np.zeros([3, 3, int(m - n)])
        for i in range(3):
            for j in range(3):
                for k in range(int(m - n)):
                    H[i, j, k] = H_n[i, j] + k * (H_m[i, j] - H_n[i, j]) / (m - n)
        return H



if __name__ == '__main__':
    pm = PitchMap()
    pm.loop()
