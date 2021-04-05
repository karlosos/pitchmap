"""
Frame Loader used for loading frame from given file
"""
import cv2
import numpy as np


class FrameLoader:
    def __init__(self, file_name):
        self.__file_name = file_name
        self.__cap = cv2.VideoCapture(f'data/{self.__file_name}')
        self.__frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def load_frame(self, frame_idx=None):
        if frame_idx is not None:
            self.set_current_frame_position(frame_idx)
        ret, frame = self.__cap.read()
        return frame

    def release(self):
        self.__cap.release()

    def get_frames_count(self):
        return int(self.__frame_count)

    def set_current_frame_position(self, frame_idx):
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def get_current_frame_position(self):
        return int(self.__cap.get(cv2.CAP_PROP_POS_FRAMES))
