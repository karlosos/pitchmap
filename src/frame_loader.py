"""
Frame Loader used for loading frame from given file
"""
import cv2


class FrameLoader:
    def __init__(self, file_name):
        self.__file_name = file_name
        self.__cap = cv2.VideoCapture(f'data/{self.__file_name}')

    def load_frame(self):
        ret, frame = self.__cap.read()
        return frame

    def release(self):
        self.__cap.release()
