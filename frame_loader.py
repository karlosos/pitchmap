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
        self.selected_frames = []

        self.select_frames_for_clustering()

    def load_frame(self):
        ret, frame = self.__cap.read()
        return frame

    def release(self):
        self.__cap.release()

    def select_frames_for_clustering(self):
        selected_frames_indexes = np.round(np.linspace(0, self.__frame_count-2, 10)).astype(int)
        for frame_idx in selected_frames_indexes:
            self.__cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frame = self.load_frame()
            self.selected_frames.append(frame.copy())

        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return self.selected_frames
