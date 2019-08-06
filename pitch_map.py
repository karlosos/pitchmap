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
        self.__video_name = 'Dynamic_Barca_Real.mp4'
        self.__fl = frame_loader.FrameLoader(self.__video_name)
        self.__display = frame_display.Display(f'PitchMap: {self.__video_name}')

        self.__trackers = cv2.MultiTracker_create()
        self.__frame_number = 0

        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        self.__tracking_method = tracking_method

    @staticmethod
    def exit_user_input():
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return True
        else:
            return False

    def loop(self):
        while True:
            frame = self.__fl.load_frame()
            frame = imutils.resize(frame, width=600)

            grass_mask = mask.grass(frame)
            edges = detect.edges_detection(grass_mask)
            lines_frame = detect.lines_detection(edges, grass_mask)

            if self.__tracking_method is not None:
                self.tracking(grass_mask)
            else:
                bounding_boxes_frame, bounding_boxes, labels = detect.players_detection(grass_mask)

            out_frame = cv2.addWeighted(grass_mask, 0.8, lines_frame, 1, 0)

            self.__display.show(out_frame)
            if self.exit_user_input():
                break

            self.__frame_number += 1

        cv2.destroyAllWindows()
        self.__fl.release()

    def tracking(self, frame):
        if self.__frame_number % 15 == 0:
            self.add_tracking_points(frame.copy())

        (success, boxes) = self.__trackers.update(frame)

        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 2)

    def add_tracking_points(self, frame):
        bounding_boxes_frame, bounding_boxes, labels = detect.players_detection(frame)
        self.__trackers = cv2.MultiTracker_create()
        for i, label in enumerate(labels):
            if label == "person":
                tracker = self.OPENCV_OBJECT_TRACKERS[self.__tracking_method]()
                self.__trackers.add(tracker, frame, tuple(bounding_boxes[i]))


if __name__ == '__main__':
    pm = PitchMap(tracking_method='kcf')
    pm.loop()
