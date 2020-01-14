import cvlib as cv
from cvlib.object_detection import draw_bbox


class PlayerDetector:
    @staticmethod
    def detect(frame):
        bbox, label, conf = cv.detect_common_objects(frame)
        return bbox, label
