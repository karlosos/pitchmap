"""
Display used for displaying images in system window
"""
import video_position_trackbar

import cv2
import imutils


class Display:
    def __init__(self, main_window_name, model_window_name, pitchmap, frame_count):
        self.__pitchmap = pitchmap

        self.__window_name = main_window_name
        self.__model_window_name = model_window_name

        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

        self.__frame_count = frame_count

        self.__video_position_trackbar = video_position_trackbar.VideoPositionTrackbar(self.__frame_count,
                                                                                       self.__pitchmap.fl)

        cv2.namedWindow(self.__window_name)
        self.__video_position_trackbar.show_trackbar(0, self.__window_name)
        cv2.setMouseCallback(self.__window_name, self.add_point_main_window)

    def show(self, frame, frame_number):
        cv2.imshow(self.__window_name, frame)
        self.__video_position_trackbar.set_trackbar(frame_number, self.__window_name)

    def show_model(self):
        cv2.imshow(self.__model_window_name, self.__pitch_model)

    def clear_model(self):
        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

    def create_model_window(self):
        cv2.namedWindow(self.__model_window_name)
        cv2.setMouseCallback(self.__model_window_name, self.add_point_model_window)

    def close_model_window(self):
        cv2.destroyWindow(self.__model_window_name)

    @staticmethod
    def close_windows():
        cv2.destroyAllWindows()

    def add_point_model_window(self, event, x, y, flags, params):
        """
        Mouse callback. For adding points of interest on model window for perspective transformation.
        """

        if event == cv2.EVENT_LBUTTONUP:
            index = self.__pitchmap.calibrator.add_point_model_window((x, y))
            if index:
                cv2.circle(self.__pitch_model, (x, y), 3, (0, 255, 0), 5)
                cv2.putText(self.__pitch_model, str(index), (x + 3, y + 3),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    def add_point_main_window(self, event, x, y, flags, params):
        """
        Mouse callback. For adding points of interest for perspective transformation.
        """
        if self.__pitchmap.calibrator.enabled:
            if event == cv2.EVENT_LBUTTONUP:
                index = self.__pitchmap.calibrator.add_point_main_window((x, y))
                print("original, model")
                print(f"Index: {index}")
                if index:
                    cv2.circle(self.__pitchmap.out_frame, (x, y), 3, (0, 255, 0), 5)
                    cv2.putText(self.__pitchmap.out_frame, str(index), (x+3, y+3),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    def add_players_to_model(self, players, player_colors):
        self.clear_model()
        for idx, player in enumerate(players):
            cv2.circle(self.__pitch_model, (int(player[0]), int(player[1])), 3, player_colors[idx], 5)

