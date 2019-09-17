import cvlib as cv
from cvlib.object_detection import draw_bbox


class PlayerDetector:
    @staticmethod
    def detect(frame):
        bbox, label, conf = cv.detect_common_objects(frame)
        out = draw_bbox(frame, bbox, label, conf)
        return out, bbox, label
