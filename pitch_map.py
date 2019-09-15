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
        self.__fl = frame_loader.FrameLoader(self.__video_name)
        self.__window_name = f'PitchMap: {self.__video_name}'
        self.calibrator = calibrator.Calibrator()
        self.__display = frame_display.Display(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               pitchmap=self)

        self.players = []
        self.players_colors = []
        self.__frame_number = 0

        self.__clf = None
        self.out_frame = None

        # Team detection initialization
        selected_frames_for_clustering = self.__fl.select_frames_for_clustering()
        self.__team_detector = team_detection.TeamDetection()
        self.__team_detector.cluster_teams(selected_frames_for_clustering)

        # Players tracking initialization
        self.__tracker = tracker.Tracker(tracking_method)

    @staticmethod
    def input_point(key):
        if key == 115:  # s key
            return True
        else:
            return False

    @staticmethod
    def input_exit(key):
        if key == 27:
            return True
        else:
            return False

    @staticmethod
    def input_transform(key):
        if key == 116:  # t key
            return True
        else:
            return False

    def loop(self):
        while True:
            if not self.calibrator.enabled:
                frame = self.__fl.load_frame()
                frame = imutils.resize(frame, width=600)

                grass_mask = mask.grass(frame)
                edges = detect.edges_detection(grass_mask)
                lines_frame = detect.lines_detection(edges, grass_mask)

                bounding_boxes_frame, bounding_boxes, labels = self.__tracker.update(grass_mask)

                self.players = []
                self.players_colors = []
                for idx, box in enumerate(bounding_boxes):
                    team_color, (x, y) = self.__team_detector.color_detection_for_player(frame, box)
                    team_id = self.__team_detector.team_detection_for_player(np.asarray(team_color))[0]

                    # TODO get team color based on team_id
                    cv2.circle(grass_mask, (x, y), 3, team_color, 5)
                    cv2.putText(grass_mask, text=str(team_id), org=(x + 3, y + 3),
                                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), lineType=1)
                    self.players.append((x, y))
                    self.players_colors.append(team_color)

                self.out_frame = cv2.addWeighted(grass_mask, 0.8, lines_frame, 1, 0)
            else:
                self.__display.show_model()

            self.__display.show(self.out_frame)

            key = cv2.waitKey(1) & 0xff
            if self.input_exit(key):
                break
            elif self.input_point(key):
                if not self.calibrator.enabled:
                    self.__display.create_model_window()
                else:
                    self.__display.close_model_window()
                self.calibrator.toggle_enabled()
            elif self.input_transform(key):
                players_2d_positions, transformed_frame = self.calibrator.calibrate(self.out_frame, self.players,
                                                                                    self.players_colors)
                self.out_frame = transformed_frame
                self.__display.add_players_to_model(players_2d_positions, self.players_colors)

            self.__frame_number += 1

        self.__display.close_windows()
        self.__fl.release()


if __name__ == '__main__':
    pm = PitchMap()
    pm.loop()
