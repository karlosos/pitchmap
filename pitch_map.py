"""
Main file of PitchMap. All process trough loading images from video to displaying 2D map.
"""
import frame_loader
import frame_display
import mask
import detect
import calibrator

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
        self.__fl = frame_loader.FrameLoader(self.__video_name)
        self.calibrator = calibrator.Calibrator()
        self.__display = frame_display.Display(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               pitchmap=self)

        self.__trackers = cv2.MultiTracker_create()
        self.__frame_number = 0

        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.__tracking_method = tracking_method
        self.out_frame = None

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

                if self.__tracking_method is not None:
                    self.tracking(grass_mask)
                else:
                    bounding_boxes_frame, bounding_boxes, labels = detect.players_detection(grass_mask)

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
                if self.calibrator.enabled and self.calibrator.get_points_count() >= 4:
                    original_points, model_points = zip(*self.calibrator.points.values())
                    original_points = np.float32(original_points)
                    model_points = np.float32(model_points)
                    rows, columns, channels = self.out_frame.shape

                    M, _ = cv2.findHomography(original_points, model_points)
                    output = cv2.warpPerspective(self.out_frame, M, (columns, rows))
                    self.out_frame = output

            self.__frame_number += 1

        self.__display.close_windows()
        self.__fl.release()

    def tracking(self, frame):
        """
        Update tracking and draw white boxes around tracking objects

        :param frame:
        """
        if self.__frame_number % 15 == 0:
            self.add_tracking_points(frame.copy())

        (success, boxes) = self.__trackers.update(frame)

        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 2)

    def add_tracking_points(self, frame):
        """
        Add points for tracking from detector
        :param frame:
        """

        bounding_boxes_frame, bounding_boxes, labels = detect.players_detection(frame)
        self.__trackers = cv2.MultiTracker_create()
        for i, label in enumerate(labels):
            if label == "person":
                tracker = self.OPENCV_OBJECT_TRACKERS[self.__tracking_method]()
                self.__trackers.add(tracker, frame, tuple(bounding_boxes[i]))


if __name__ == '__main__':
    pm = PitchMap()
    pm.loop()
