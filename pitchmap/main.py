"""
Main file of PitchMap. All process trough loading images from video to displaying 2D map.
"""
from pitchmap.detect import team
from pitchmap.cache_loader import clustering_model, pickler, players_detector
from pitchmap.frame import loader
from pitchmap.segmentation import mask
from pitchmap.players import structure
from pitchmap.detect import players
from pitchmap.homography import calibrator, calibrator_interactor
from pitchmap import gui

import imutils
import cv2
import numpy as np
import threading
import time
import os


class PitchMap:
    def __init__(self, tracking_method=None):
        """
        :param tracking_method: if left default (None) then there's no tracking and detection
        is performed in every frame
        """
        # display = gui.high.Display
        display = gui.pygame.Display
        player_list = structure.PlayersListComplex
        # player_list = player.PlayersListSimple
        calib_interactor = calibrator_interactor.CalibrationInteractorMiddlePoint
        # calib_interactor = calibrator_interactor.CalibrationInteractorSimple

        self.video_name = 'baltyk_starogard_1.mp4'
        self.__window_name = f'PitchMap: {self.video_name}'

        self.fl = loader.FrameLoader(self.video_name)
        self.calibrator = calibrator.Calibrator()
        self.__display = display(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                 pitchmap=self, frame_count=self.fl.get_frames_count())
        self.players_list = player_list(frames_length=self.fl.get_frames_count())
        self.out_frame = None

        # Team detection initialization
        self.__team_detector = team.TeamDetection()
        clf_model_loader = clustering_model.ClusteringModelLoader(self.__team_detector,
                                                                  self.fl, self.video_name)
        clf_model_loader.generate_clustering_model()

        # Players tracking initialization
        frames_length = self.fl.get_frames_count()
        self.players_detector = players.PlayerDetector()
        self.players_detector.loader = players_detector.PlayersDetectorLoader(frames_length, self.video_name)

        self.__interpolation_mode = False

        self.team_colors = [(35, 117, 250), (250, 46, 35), (255, 48, 241)]

        self.__detection_thread = None
        self.__calibration_interactor = calib_interactor(pitch_map=self, calibrator=self.calibrator,
                                                         frame_loader=self.fl)
        self.transforming_flag = False
        self.detecting_flag = False

        player_list_class_name = self.players_list.__class__.__name__
        calib_inter_class_name = self.__calibration_interactor.__class__.__name__
        self.__save_data_path = f'data/cache/{self.video_name}_{player_list_class_name}_{calib_inter_class_name}.pik'
        self.bootstrap()

        self.pause = False

    def frame_loading(self):
        frame = self.fl.load_frame()
        frame_number = self.fl.get_current_frame_position()
        print(f"Frame: {frame_number}")
        try:
            frame = imutils.resize(frame, width=600)
        except AttributeError:
            return

        grass_mask = mask.grass(frame)
        # edges = detect.edges_detection(grass_mask)
        # lines_frame = detect.lines_detection(edges, grass_mask)

        bounding_boxes = []

        if self.detecting_flag:
            bounding_boxes_frame, bounding_boxes, labels = self.players_detector.detect(grass_mask, frame_number)
            player_indices = [i for i, x in enumerate(labels) if x == 'person']
            players_bounding_boxes = []
            bounding_boxes_length = len(bounding_boxes)
            for index in sorted(player_indices, reverse=True):
                if index < bounding_boxes_length:
                    players_bounding_boxes.append(bounding_boxes[index])
            bounding_boxes = players_bounding_boxes

        self.draw_bounding_boxes(frame, grass_mask, bounding_boxes)
        # self.out_frame = cv2.addWeighted(grass_mask, 0.8, lines_frame, 1, 0)
        if self.transforming_flag:
            self.out_frame = self.transform_frame(grass_mask, self.fl.get_current_frame_position())
        else:
            self.out_frame = grass_mask

    def loop(self):
        self.frame_loading()
        previous = time.perf_counter()
        while True:
            # user input
            is_exit = not self.__display.input_events()
            if is_exit:
                break
            if not self.pause:
                frame_number = self.fl.get_current_frame_position()
                current = time.perf_counter()

                # update
                if not self.calibrator.enabled:
                    if self.__detection_thread is None or not self.__detection_thread.is_alive():
                        if current - previous > 0.04:
                            self.__detection_thread = threading.Thread(target=self.frame_loading)
                            self.__detection_thread.start()
                            previous = current

                players_2d_positions, colors = self.get_players_positions_on_model(frame_number)
                self.__display.show_model(players_2d_positions, colors)
            self.__display.show(self.out_frame, frame_number)
            self.__display.update()

        self.__display.close_windows()
        self.fl.release()
        self.teardown()

    def teardown(self):
        homographies = []
        for i in range(self.fl.get_frames_count()):
            homographies.append(self.__calibration_interactor.get_homography(i))

        pickler.pickle_data([self.players_list.players, self.__calibration_interactor.homographies, homographies],
                            self.__save_data_path + "two_halves")
        print(f"Saved data to: {self.__save_data_path}")
        self.players_detector.loader.save_data()

    def bootstrap(self):
        file_exists = os.path.isfile(self.__save_data_path)
        if file_exists:
            players, h, _ = pickler.unpickle_data(self.__save_data_path)
            self.players_list.players = players
            self.__calibration_interactor.homographies = h
            print(f"Loaded data from: {self.__save_data_path}")
        else:
            print("No data to load")

    def disable_interpolation(self):
        if self.fl.get_current_frame_position() > self.calibrator.stop_calibration_frame_index:
            self.__interpolation_mode = False
            self.calibrator.clear_interpolation()

    def draw_bounding_boxes(self, frame, grass_mask, bounding_boxes):
        current_frame_number = self.fl.get_current_frame_position()
        if self.detecting_flag:
            self.players_list.clear(current_frame_number)
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

    def transform_frame(self, frame, frame_idx):
        rows, columns, channels = frame.shape
        h = self.__calibration_interactor.get_homography(frame_idx)
        transformed_frame = cv2.warpPerspective(frame, h, (columns, rows))
        return transformed_frame

    def start_calibration(self):
        return self.__calibration_interactor.start_calibration()

    def perform_transform(self):
        self.__calibration_interactor.perform_transform(players_list=self.players_list, team_colors=self.team_colors)

    def accept_transform(self):
        self.__calibration_interactor.accept_transform()

    def input_test(self):
        self.fl.set_current_frame_position(99)

    def add_players_to_model(self, players_2d_positions, colors):
        self.__display.add_players_to_model(players_2d_positions, colors)

    def set_transforming_flag(self, state=False):
        self.transforming_flag = state

    def create_model_window(self):
        self.__display.create_model_window()
        self.__display.clear_model()
        self.__display.show_model()

    def load_frame(self):
        self.__detection_thread = threading.Thread(target=self.frame_loading)
        self.__detection_thread.start()

    def toggle_detecting(self):
        self.detecting_flag = not self.detecting_flag
        return self.detecting_flag

    def toggle_transforming(self):
        self.transforming_flag = not self.transforming_flag
        return self.transforming_flag

    def get_players_positions_on_model(self, frame_idx):
        players_2d_positions = []
        colors = []
        has_players_positions = self.players_list.is_frame_populated(frame_idx)
        has_homography = self.__calibration_interactor.is_homography_exist(frame_idx)
        if has_players_positions and has_homography:
            players = self.players_list.get_players_positions_from_frame(frame_number=frame_idx)
            team_ids = self.players_list.get_players_team_ids_from_frame(frame_number=frame_idx)
            colors = list(map(lambda x: self.team_colors[x], team_ids))
            current_frame_homography = self.__calibration_interactor.get_homography(frame_idx)
            players_2d_positions = self.calibrator.transform_to_2d(players, current_frame_homography)

        return players_2d_positions, colors

    def toggle_pause(self):
        self.pause = not self.pause


if __name__ == '__main__':
    pm = PitchMap()
    pm.loop()
