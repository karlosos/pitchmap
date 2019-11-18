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
import player
from gui_pygame import display

import imutils
import cv2
import numpy as np
import threading
import time


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
        # self.__display = frame_display.Display(main_window_name=self.__window_name,
        #                                              model_window_name="2D Pitch Model",
        #                                              pitchmap=self, frame_count=self.fl.get_frames_count())
        self.__display = display.PyGameDisplay(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               pitchmap=self, frame_count=self.fl.get_frames_count())

        #self.players_list = player.PlayersListSimple(frames_length=self.fl.get_frames_count())
        self.players_list = player.PlayersListComplex(frames_length=self.fl.get_frames_count())

        self.out_frame = None

        # Team detection initialization
        selected_frames_for_clustering = self.fl.select_frames_for_clustering()
        self.__team_detector = team_detection.TeamDetection()
        self.__team_detector.cluster_teams(selected_frames_for_clustering)

        # Players tracking initialization
        self.__tracker = tracker.Tracker(tracking_method)

        self.__interpolation_mode = False

        self.team_colors = [(35, 117, 250), (250, 46, 35), (255, 48, 241)]

        self.__detection_thread = None

    def frame_loading(self, detecting=False, interpolating=False):
        frame = self.fl.load_frame()
        frame = imutils.resize(frame, width=600)

        grass_mask = mask.grass(frame)
        # edges = detect.edges_detection(grass_mask)
        # lines_frame = detect.lines_detection(edges, grass_mask)

        bounding_boxes = []
        if detecting or interpolating:
            bounding_boxes_frame, bounding_boxes, labels = self.__tracker.update(grass_mask)

        self.players_list.clear()

        self.draw_bounding_boxes(frame, grass_mask, bounding_boxes)

        # self.out_frame = cv2.addWeighted(grass_mask, 0.8, lines_frame, 1, 0)
        self.out_frame = grass_mask

    def loop(self):
        self.frame_loading()
        previous = time.perf_counter()
        while True:
            current = time.perf_counter()
            # user input
            is_exit = not self.__display.input_events()
            if is_exit:
                break
            # update
            if not self.calibrator.enabled:
                if self.__detection_thread is None or not self.__detection_thread.is_alive():
                    if current - previous > 0.04:
                        self.__detection_thread = threading.Thread(target=self.frame_loading,
                                                                   args=(self.calibrator.enabled, self.__interpolation_mode))
                        self.__detection_thread.start()
                        previous = current

            if self.__interpolation_mode:
                self.disable_interpolation()

            self.__display.show_model()

            self.__display.show(self.out_frame, self.fl.get_current_frame_position())

            self.__display.update()

        self.__display.close_windows()
        self.fl.release()

    def disable_interpolation(self):
        if self.fl.get_current_frame_position() > self.calibrator.stop_calibration_frame_index:
            self.__interpolation_mode = False
            self.calibrator.clear_interpolation()

    def draw_bounding_boxes(self, frame, grass_mask, bounding_boxes):
        current_frame_number = self.fl.get_current_frame_position()
        for idx, box in enumerate(bounding_boxes):
            player_color, (x, y) = self.__team_detector.color_detector.color_detection_for_player(frame, box)
            team_id = self.__team_detector.team_detection_for_player(np.asarray(player_color))[0]
            player = self.players_list.assign_player(position=(x, y), color=team_id,
                                                     frame_number=current_frame_number)
            calculated_team = player.calculate_real_color()
            team_color = self.team_colors[calculated_team]

            cv2.circle(grass_mask, (x, y), 3, team_color, 5)
            cv2.putText(grass_mask, text=f"{player.id}:{team_id}", org=(x, y + 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), lineType=1)

    def start_calibration(self):
        if not self.calibrator.enabled:
            self.__display.create_model_window()
            self.__display.clear_model()
            self.__display.show_model()
        else:
            self.__display.close_model_window()
        status = self.calibrator.toggle_enabled()
        self.__detection_thread = threading.Thread(target=self.frame_loading,
                                                   args=(self.calibrator.enabled,))
        self.__detection_thread.start()
        return status

    def perform_transform(self):
        if self.calibrator.can_perform_calibrate():
            if self.calibrator.stop_calibration_H is None:
                players = self.players_list.get_players_positions_from_frame(
                    frame_number=self.fl.get_current_frame_position())
                team_ids = self.players_list.get_players_team_ids_from_frame(
                    frame_number=self.fl.get_current_frame_position())
                colors = list(map(lambda x: self.team_colors[x], team_ids))
                players_2d_positions, transformed_frame, H = self.calibrator.calibrate(self.out_frame, players,
                                                                                       colors)
                self.out_frame = transformed_frame
                self.__display.add_players_to_model(players_2d_positions, colors)

                if self.calibrator.start_calibration_H is None:
                    print("Start calibration")
                    self.calibrator.start_calibration(H, self.fl.get_current_frame_position())
                elif self.calibrator.stop_calibration_H is None:
                    print("Stop calibration")
                    self.calibrator.end_calibration(H, self.fl.get_current_frame_position())
            else:
                print("Interpolation mode start")
                self.__interpolation_mode = True
                self.calibrator.enabled = False
                self.fl.set_current_frame_position(self.calibrator.start_calibration_frame_index)

    def input_test(self):
        self.fl.set_current_frame_position(99)


if __name__ == '__main__':
    pm = PitchMap()
    pm.loop()
