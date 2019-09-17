import plotting
import players_detector
import mask

from sklearn.cluster import AgglomerativeClustering
from sklearn import tree
import imutils
import cv2
import numpy as np


class TeamDetection:
    def __init__(self, plot=False):
        self.__clf = None
        self.__plot = plot

    def cluster_teams(self, selected_frames):
        extracted_player_colors = self.extract_player_colors(selected_frames)

        clust = AgglomerativeClustering(n_clusters=3).fit(extracted_player_colors)
        if self.__plot:
            plotting.plot_colors(extracted_player_colors, clust.labels_)

        self.__clf = tree.DecisionTreeClassifier()
        self.__clf = self.__clf.fit(extracted_player_colors, clust.labels_)

    def extract_player_colors(self, frames):
        extracted_player_colors = []
        player_detector = players_detector.PlayerDetector()
        for frame in frames:
            frame = imutils.resize(frame, width=600)
            grass_mask = mask.grass(frame)
            _, bounding_boxes, labels = player_detector.detect(grass_mask)
            bounding_boxes = self.serialize_bounding_boxes(bounding_boxes)
            for idx, box in enumerate(bounding_boxes):
                if labels[idx] == 'person':
                    team_color, _ = self.color_detection_for_player(frame, box)
                    extracted_player_colors.append(team_color)
        return extracted_player_colors

    @staticmethod
    def color_detection_for_player(frame, box):
        x = int((box[0] + box[2]) / 2)
        y = box[3]
        player_crop = frame[box[1]:box[3], box[0]:box[2], :]
        grass_mask_player_crop = cv2.bitwise_not(mask.grass(player_crop, True))
        player_crop = cv2.bitwise_and(player_crop, player_crop,
                                      mask=grass_mask_player_crop)
        mean_color = cv2.mean(player_crop, mask=grass_mask_player_crop)
        return mean_color, (x, y)

    def team_detection_for_player(self, color):
        return self.__clf.predict(color.reshape(1, -1))

    @staticmethod
    def serialize_bounding_boxes(bounding_boxes):
        bounding_boxes = np.where(np.asarray(bounding_boxes) < 0, 0,
                                  bounding_boxes)
        return bounding_boxes