"""
Main file of PitchMap. All process trough loading images from video to displaying 2D map.
"""
import frame_loader
import frame_display
import mask
import detect

import imutils
import cv2


class PitchMap:
    def __init__(self, tracking_method=None):
        """
        :param tracking_method: if left default (None) then there's no tracking and detection
        is performed in every frame
        """
        self.__video_name = 'Dynamic_Barca_Real.mp4'
        self.__window_name = f'PitchMap: {self.__video_name}'
        self.__fl = frame_loader.FrameLoader(self.__video_name)
        self.__display = frame_display.Display(self.__window_name)

        self.__trackers = cv2.MultiTracker_create()
        self.__frame_number = 0

        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.__tracking_method = tracking_method

        cv2.namedWindow(self.__window_name)
        cv2.setMouseCallback(self.__window_name, self.add_point)

        self.__pause = False
        self.__out_frame = None

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

    def loop(self):
        while True:
            if not self.__pause:
                frame = self.__fl.load_frame()
                frame = imutils.resize(frame, width=600)

                grass_mask = mask.grass(frame)
                edges = detect.edges_detection(grass_mask)
                lines_frame = detect.lines_detection(edges, grass_mask)

                if self.__tracking_method is not None:
                    self.tracking(grass_mask)
                else:
                    bounding_boxes_frame, bounding_boxes, labels = detect.players_detection(grass_mask)

                self.__out_frame = cv2.addWeighted(grass_mask, 0.8, lines_frame, 1, 0)

            self.__display.show(self.__out_frame)

            key = cv2.waitKey(1) & 0xff
            if self.input_exit(key):
                break
            elif self.input_point(key):
                self.__pause = not self.__pause

            self.__frame_number += 1

        cv2.destroyAllWindows()
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

    def add_point(self, event, x, y, flags, params):
        """
        Mouse callback. For adding points of interest for perspective transformation.
        """

        if self.__pause:
            if event == cv2.EVENT_LBUTTONUP:
                cv2.circle(self.__out_frame, (x, y), 5, (0, 255, 0), 2)


if __name__ == '__main__':
    pm = PitchMap()
    pm.loop()
