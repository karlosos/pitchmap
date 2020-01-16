from pitchmap.detect import players

import cv2


class Tracker:
    def __init__(self, tracking_method):
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.__tracking_method = tracking_method
        self.__trackers = cv2.MultiTracker_create()
        self.players_detector = players.PlayerDetector()
        self.__frame_number = 0

    def update(self, frame):
        if self.is_tracking_enabled():
            self.tracking(frame)
            # TODO return bounding_boxes_frame, bounding_boxes, labels
        else:
            frame, bounding_boxes, labels = self.players_detector.detect(frame)
            return frame, bounding_boxes, labels
        self.__frame_number += 1

    def tracking(self, frame):
        """
        Update tracking and draw white boxes around tracking objects

        :param frame:
        """
        if self.__frame_number % 15 == 0:
            self.add_tracking_points(frame.copy())

        (success, boxes) = self.__trackers.update(frame)
        self.draw_bounding_boxes(frame, boxes)

    @staticmethod
    def draw_bounding_boxes(frame, boxes):
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 2)

    def add_tracking_points(self, frame):
        """
        Add points for tracking from detector
        :param frame:
        """
        bounding_boxes, labels = self.players_detector.detect(frame)
        self.__trackers = cv2.MultiTracker_create()
        for i, label in enumerate(labels):
            if label == "person":
                tracker = self.OPENCV_OBJECT_TRACKERS[self.__tracking_method]()
                self.__trackers.add(tracker, frame, tuple(bounding_boxes[i]))

    def is_tracking_enabled(self):
        return self.__tracking_method is not None
