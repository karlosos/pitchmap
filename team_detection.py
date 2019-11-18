import plotting
import players_detector
import mask

from sklearn.cluster import AgglomerativeClustering
from sklearn import tree
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt


class TeamDetection:
    def __init__(self, plot=False):
        self.__clf = None
        self.__plot = plot
        self.color_detector = ColorDetectorUpperHalf()

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
                    team_color, _ = self.color_detector.color_detection_for_player(frame, box)
                    extracted_player_colors.append(team_color)
        return extracted_player_colors

    def team_detection_for_player(self, color):
        return self.__clf.predict(color.reshape(1, -1))

    @staticmethod
    def serialize_bounding_boxes(bounding_boxes):
        bounding_boxes = np.where(np.asarray(bounding_boxes) < 0, 0,
                                  bounding_boxes)
        return bounding_boxes


class ColorDetectorEntirePlayer:
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


class ColorDetectorUpperHalf:
    @staticmethod
    def color_detection_for_player(frame, box):
        x = int((box[0] + box[2]) / 2)
        y = box[3]
        half_y = int((box[3]-box[1])/2)
        player_crop = frame[box[1]:box[3]-half_y, box[0]:box[2], :]
        grass_mask_player_crop = cv2.bitwise_not(mask.grass(player_crop, True))
        player_crop = cv2.bitwise_and(player_crop, player_crop,
                                      mask=grass_mask_player_crop)
        mean_color = cv2.mean(player_crop, mask=grass_mask_player_crop)
        color_row = np.zeros(player_crop.shape, dtype=int)
        color_row[:, :, 0] = mean_color[0]
        color_row[:, :, 1] = mean_color[1]
        color_row[:, :, 2] = mean_color[2]
        player_with_color = np.vstack((player_crop, color_row))
        # plt.imshow(player_with_color)
        # plt.show()
        color = np.uint8([[[mean_color[0], mean_color[1], mean_color[2]]]])
        mean_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).tolist()[0][0]
        return mean_color, (x, y)


class ColorDetectorLowerHalf:
    @staticmethod
    def color_detection_for_player(frame, box):
        x = int((box[0] + box[2]) / 2)
        y = box[3]
        half_y = int((box[3]-box[1])/2)
        player_crop = frame[box[1]+half_y:box[3], box[0]:box[2], :]
        grass_mask_player_crop = cv2.bitwise_not(mask.grass(player_crop, True))
        player_crop = cv2.bitwise_and(player_crop, player_crop,
                                      mask=grass_mask_player_crop)
        mean_color = cv2.mean(player_crop, mask=grass_mask_player_crop)
        color_row = np.zeros(player_crop.shape, dtype=int)
        color_row[:, :, 0] = mean_color[0]
        color_row[:, :, 1] = mean_color[1]
        color_row[:, :, 2] = mean_color[2]
        player_with_color = np.vstack((player_crop, color_row))
        player_with_color = player_with_color.astype(np.uint8)
        color = np.uint8([[[mean_color[0], mean_color[1], mean_color[2]]]])
        mean_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).tolist()[0][0]
        return mean_color, (x, y)


class ColorDetectorTwoHalves:
    @staticmethod
    def color_detection_for_player(frame, box):
        x = int((box[0] + box[2]) / 2)
        y = box[3]
        half_y = int((box[3]-box[1])/2)
        # upper
        player_crop_upper = frame[box[1]+half_y:box[3], box[0]:box[2], :]
        grass_mask_player_crop_upper = cv2.bitwise_not(mask.grass(player_crop_upper, True))
        player_crop_upper = cv2.bitwise_and(player_crop_upper, player_crop_upper,
                                      mask=grass_mask_player_crop_upper)
        mean_color_upper = cv2.mean(player_crop_upper, mask=grass_mask_player_crop_upper)
        color = np.uint8([[[mean_color_upper[0], mean_color_upper[1], mean_color_upper[2]]]])
        mean_color_upper = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).tolist()[0][0]

        # lower
        player_crop_lower = frame[box[1]+half_y:box[3], box[0]:box[2], :]
        grass_mask_player_crop_lower = cv2.bitwise_not(mask.grass(player_crop_lower, True))
        player_crop_lower = cv2.bitwise_and(player_crop_lower, player_crop_lower,
                                      mask=grass_mask_player_crop_lower)
        mean_color_lower = cv2.mean(player_crop_lower, mask=grass_mask_player_crop_lower)
        color = np.uint8([[[mean_color_lower[0], mean_color_lower[1], mean_color_lower[2]]]])
        mean_color_lower = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).tolist()[0][0]

        mean_color_two_halves = mean_color_upper + mean_color_lower
        return mean_color_two_halves, (x, y)
