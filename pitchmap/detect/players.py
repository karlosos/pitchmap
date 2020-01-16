import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2


class PlayerDetector:
    def __init__(self):
        self.loader = None

    def detect(self, frame, frame_number=None):
        if frame_number is None:
            bbox, label, conf = cv.detect_common_objects(frame)
            return frame, bbox, label
        if self.loader is not None:
            if self.loader.get_detections(frame_number) is None:
                bbox, label, conf = cv.detect_common_objects(frame)
                self.loader.set_detections(detections=(bbox, label), frame_number=frame_number)
                return frame, bbox, label
            else:
                bbox, label = self.loader.get_detections(frame_number)
                return frame, bbox, label
