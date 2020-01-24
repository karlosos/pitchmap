import os
from pitchmap.cache_loader import pickler


class ClusteringModelLoader:
    def __init__(self, team_detector, frame_loader, video_name):
        self.__team_detector = team_detector
        self.fl = frame_loader

        color_detector_name = self.__team_detector.color_detector.__class__.__name__
        team_detection_file_name = f"data/cache/{video_name}_{color_detector_name}.pik"
        self.__file_name = team_detection_file_name

    def generate_clustering_model(self):
        team_detection_file_name = self.__file_name
        file_exists = os.path.isfile(team_detection_file_name)

        if file_exists:
            print(f"loading from {team_detection_file_name}")
            clf = pickler.unpickle_data(team_detection_file_name)
            self.__team_detector.set_clf(clf)
        else:
            print("saving")
            selected_frames_for_clustering = self.fl.select_frames_for_clustering()
            self.__team_detector.cluster_teams(selected_frames_for_clustering)
            pickler.pickle_data(self.__team_detector.get_clf(), team_detection_file_name)
