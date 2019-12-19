from manual_tracking.frame import loader
from manual_tracking.gui import display
import manual_tracking.players as players
import manual_tracking.calibrator as calibrator
from pitchmap.cache_loader import pickler
import os

import imutils
import cv2


class ManualTracker:
    def __init__(self):
        self.video_name = 'baltyk_starogard_1.mp4'
        self.__window_name = f'PitchMap: {self.video_name}'

        self.fl = loader.FrameLoader(self.video_name)
        self.__display = display.PyGameDisplay(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               main_object=self, frame_count=self.fl.get_frames_count())
        self.__players_list = players.PlayersList(self.fl.get_frames_count())
        self.out_frame = None
        self.transformed_frame = None
        self.current_player = None
        self.calibrator = calibrator.Calibrator()
        self.homographies = {}
        self.bootstrap()

    def add_player(self, position):
        player = self.__players_list.create_player(position, self.fl.get_current_frame_position())
        self.current_player = player
        return player

    def change_player_id(self, player_id):
        self.current_player.id = player_id

    def change_player_color(self, color):
        self.current_player.color = color

    def calibration(self):
        state = self.calibrator.toggle_enabled()
        return state

    def find_homography(self):
        transformed_frame, H = self.calibrator.find_homography(self.out_frame, self.__display.pitch_model)
        self.transformed_frame = transformed_frame
        pitch_model = self.__display.pitch_model
        self.transformed_frame = cv2.addWeighted(self.transformed_frame, 0.7, pitch_model, 0.3, 0.0)
        self.homographies[self.fl.get_current_frame_position()] = H

    def reset_calibration_points(self):
        self.calibrator.clear_points()
        self.__display.calibration_circles = []

    def delete_player(self):
        frame_id = self.current_player.frame_number
        self.__players_list.players[frame_id].remove(self.current_player)
        self.current_player = None

    def load_next_frame(self):
        frame = self.fl.load_frame()
        try:
            frame = imutils.resize(frame, width=600)
        except AttributeError:
            return
        self.out_frame = frame
        self.transform_frame()
        self.__display.refresh_points()

    def transform_frame(self):
        frame_number = self.fl.get_current_frame_position()
        H = self.homographies.get(frame_number, None)
        if H is not None:
            self.transformed_frame = self.calibrator.transform_frame(self.out_frame, H, self.__display.pitch_model)
            pitch_model = self.__display.pitch_model
            self.transformed_frame = cv2.addWeighted(self.transformed_frame, 0.5, pitch_model, 0.5, 0.0)
        else:
            self.transformed_frame = self.out_frame

    def load_previous_frame(self):
        frame_number = self.fl.get_current_frame_position()
        self.fl.set_current_frame_position(frame_number-2)
        self.load_next_frame()

    def loop(self):
        self.load_next_frame()
        while True:
            is_exit = not self.__display.input_events()
            if is_exit:
                break

            frame_number = self.fl.get_current_frame_position()
            self.__display.show(self.out_frame, self.transformed_frame, frame_number)
            last_frame_players = []
            if frame_number > 0:
                last_frame_players = self.__players_list.players[frame_number-1]
            current_frame_players = self.__players_list.players[frame_number]
            self.__display.show_model(current_frame_players, last_frame_players)
            self.__display.update()

    def teardown(self):
        data_path = f"data/cache/{self.video_name}_manual_tracking.pik"
        pickler.pickle_data([self.__players_list, self.homographies, self.calibrator],
                            data_path)
        print(f"Saved data to: {data_path}")

    def bootstrap(self):
        data_path = f"data/cache/{self.video_name}_manual_tracking.pik"
        file_exists = os.path.isfile(data_path)
        if file_exists:
            players_list, homographies, calib = pickler.unpickle_data(data_path)
            self.__players_list = players_list
            self.homographies = homographies
            self.calibrator = calib
            print(f"Loaded data from: {data_path}")
        else:
            print("No data to load")


if __name__ == '__main__':
    mt = ManualTracker()
    mt.loop()
    mt.teardown()
