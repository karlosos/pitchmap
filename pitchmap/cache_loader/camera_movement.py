import os
from pitchmap.cache_loader import pickler


class CameraMovementLoader:
    def __init__(self, camera_movement_analyser, video_name):
        self.__camera_movement_analyser = camera_movement_analyser

        camera_movement_name = self.__camera_movement_analyser.__class__.__name__
        team_detection_file_name = f"data/cache/{video_name}_{camera_movement_name}.pik"
        self.__file_name = team_detection_file_name

    def load(self):
        file_name = self.__file_name
        file_exists = os.path.isfile(file_name)

        if file_exists:
            print("loading camera movement")
            x_cum_sum = pickler.unpickle_data(file_name)
            self.__camera_movement_analyser.x_cum_sum = x_cum_sum
            self.__camera_movement_analyser.calculate_characteristic_points()
            return True
        else:
            return False

    def save(self):
        file_name = self.__file_name
        print("saving camera movement")
        pickler.pickle_data(self.__camera_movement_analyser.x_cum_sum, file_name)
