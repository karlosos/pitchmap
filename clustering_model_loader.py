import os
import pickle


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
            print("loading")
            clf = self.unpickle(team_detection_file_name)
            self.__team_detector.set_clf(clf)
        else:
            print("saving")
            selected_frames_for_clustering = self.fl.select_frames_for_clustering()
            self.__team_detector.cluster_teams(selected_frames_for_clustering)
            self.pickle(self.__team_detector.get_clf(), team_detection_file_name)

    @staticmethod
    def unpickle(file_name):
        f = open(file_name, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def pickle(data, file_name):
        print(file_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        f = open(file_name, 'wb')
        pickle.dump(data, f)
        f.close()
