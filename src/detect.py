"""
Detects players from frame

https://github.com/arunponnusamy/cvlib
"""
import cvlib as cv
from cvlib.object_detection import draw_bbox


def players(frame):
    bbox, label, conf = cv.detect_common_objects(frame)
    out = draw_bbox(frame, bbox, label, conf)
    return out, bbox, label
