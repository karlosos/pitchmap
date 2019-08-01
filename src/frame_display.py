"""
Display used for displaying images in system window
"""
import cv2


class Display:
    def __init__(self, window_name):
        self.__window_name = window_name

    def show(self, frame):
        cv2.imshow(self.__window_name, frame)
