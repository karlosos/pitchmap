"""
Program testowy to trackingu obiektów. Można określić czy chcemy używać ręcznego oznaczania
ROI czy automatycznie wykrywać piłkarzy przez cvlib.

Wnioski:
- dodawanie trackerów trwa długo: około 0.3 sekundy na jeden obiekt
    - przy założeniu że piłkarzy jest 18 jest to 5 sekund na samo dodanie trackerów
- update trackerów trwa szybciej, ale też nie jest zadowalająco szybkie
- rozjeżdzają się prostokąty
    - dla testy zwiększono obszar prostokątu, co daje lepsze rezultaty

Używanie:
- zmień flagę MANUAL i BIGGER_BBOXES
- automatyczne:
    działa bez inputu człowieka
- manualne
    1. zaznacz piłkarza
    2. kliknij spację lub enter aby potwierdzić zaznaczenie
    3. powtórz kroki aby oznaczyć wszystkich piłkarzy
    4. esc aby zatwierdzić selekcję
"""

import imutils
import cv2
import sys
import cvlib as cv
from cvlib.object_detection import draw_bbox
import time

MANUAL = False
BIGGER_BBOXES = False

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()
vs = cv2.VideoCapture("data/baltyk_starogard_1.mp4")

sleep_time = 0


def detect(frame):
    bbox, label, conf = cv.detect_common_objects(frame)
    out = draw_bbox(frame, bbox, label, conf)
    return out, bbox, label


def select_detection(frame):
    create_t1 = time.perf_counter()
    trackers = cv2.MultiTracker_create()
    create_t2 = time.perf_counter()
    print(f"create tracker: {create_t2-create_t1}")
    out, boxes, labels = detect(frame)
    cv2.imwrite("test.png", out)
    add_t1 = time.perf_counter()
    for idx, box in enumerate(boxes):
        if labels[idx] == 'person':
            x1, y1, x2, y2 = box
            if BIGGER_BBOXES:
                roi = (x1-5, y1-5, x2-x1+10, y2-y1+10)
            else:
                roi = (x1, y1, x2 - x1, y2 - y1)
            tracker = OPENCV_OBJECT_TRACKERS['boosting']()
            try:
                trackers.add(tracker, frame, tuple(roi))
            except:
                print(f"Couldn't add tracker for {roi}")
    add_t2 = time.perf_counter()
    print(f"add trackers: {add_t2 - add_t1}, avg: {(add_t2 - add_t1)/len(boxes)}")
    return trackers


def select_manual(frame):
    trackers = cv2.MultiTracker_create()
    boxes = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)
    add_t1 = time.perf_counter()
    for box in boxes:
        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
        trackers.add(tracker, frame, tuple(box))
    add_t2 = time.perf_counter()
    print(f"add trackers: {add_t2 - add_t1}, avg: {(add_t2 - add_t1)/len(boxes)}")
    return trackers


while True:
    frame_number = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
    time.sleep(sleep_time)
    if not vs.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = vs.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    clear_frame = frame.copy()

    (success, boxes) = trackers.update(frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("o"):
        sleep_time = 1
    if key == ord("p"):
        sleep_time = 0

    if frame_number % 20 == 0:
        if MANUAL:
            trackers = select_manual(frame)
        else:
            trackers = select_detection(clear_frame)

    elif key == ord("q"):
        break

vs.release()

cv2.destroyAllWindows()
