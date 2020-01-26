import os
from pitchmap.cache_loader import pickler


class PlayersDetectorLoader:
    def __init__(self, frames_length, video_name):
        file_name = f"data/cache/{video_name}_detected.pik"
        self.__file_name = file_name
        self.frames = [[] for _ in range(frames_length + 1)]
        self.load_data()

    def load_data(self):
        file_exists = os.path.isfile(self.__file_name)
        if file_exists:
            print("loading detected frames")
            frames = pickler.unpickle_data(self.__file_name)
            self.frames = frames
        else:
            print("no file with detected frames")

    def save_data(self):
        print("saved data bounding boxes")
        pickler.pickle_data(self.frames, self.__file_name)

    def get_detections(self, frame_number):
        if self.frames[int(frame_number)]:
            return self.frames[int(frame_number)]
        else:
            return None

    def set_detections(self, detections, frame_number):
        print("set detections")
        self.frames[frame_number] = detections
