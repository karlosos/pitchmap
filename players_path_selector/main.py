from manual_tracking.frame import loader
from players_path_selector.gui import display
from pitchmap.players import structure
from pitchmap.cache_loader import pickler
import os

import imutils
import numpy as np
from numpy.linalg import inv

FAST_ADDING = False


class PlayersPathSelector:
    def __init__(self):
        self.video_name = 'baltyk_kotwica_1.mp4'
        self.file_detected_data = "data/cache/baltyk_kotwica_1.mp4_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"
        self.file_manual_data = "data/cache/baltyk_kotwica_1.mp4_manual_tracking.pik"
        self.__window_name = f'Path selector: {self.video_name}'

        self.fl = loader.FrameLoader(self.video_name)
        self.__display = display.PyGameDisplay(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               main_object=self, frame_count=self.fl.get_frames_count())
        self.out_frame = None
        self.current_player = None
        self.players_detected = []
        self.players_manual = []
        self.homographies_detected = []
        self.homographies_manual = []

        self.load_data()

        self.bootstrap()

    def load_data(self):
        players_detected, _, homographies_detected = pickler.unpickle_data(self.file_detected_data)
        players_list_manual, homographies_manual, _ = pickler.unpickle_data(self.file_manual_data)

        self.homographies_detected = {}
        players_detected_transformed = []
        length_homographies_detected = len(homographies_detected)
        for i, players in enumerate(players_detected):
            players_positions = self.get_players_positions_from_frame(players_detected, i)
            homography = homographies_detected[i] if i < length_homographies_detected else homographies_detected[-1]
            self.homographies_detected[i] = homography

            players_2d_positions = self.transform_bulk_players_to_2d(players_positions, homography)
            players_colors = self.get_players_team_ids_from_frame(players_detected, i)
            players_ids = self.get_players_ids(players_detected, i)

            players_in_frame = []
            for i, position in enumerate(players_2d_positions):
                player = structure.PlayerSimple((position[0], position[1]), players_colors[i])
                player.id = players_ids[i]
                players_in_frame.append(player)
            players_detected_transformed.append(players_in_frame)

        self.players_detected = players_detected_transformed
        self.players_manual = players_list_manual.players

        self.homographies_manual = homographies_manual

    def get_players_positions_from_frame(self, players, frame_number):
        try:
            players = players[frame_number]
        except IndexError:
            players = []

        if players:
            if type(players) is dict:
                players = list(players.values())
            positions = [player.position for player in players]
        else:
            positions = []

        return positions

    def get_players_ids(self, players, frame_number):
        try:
            players = players[frame_number]
        except IndexError:
            players = []

        if players:
            if type(players) is dict:
                players = list(players.values())
            ids = list(map(lambda player: player.id, players))
        else:
            ids = []
        return ids

    def get_players_team_ids_from_frame(self, players, frame_number):
        try:
            players = players[frame_number]
        except IndexError:
            players = []

        if players:
            if type(players) is dict:
                players = list(players.values())
            colors = list(map(lambda player: player.calculate_real_color(), players))
        else:
            colors = []
        return colors

    def load_next_frame(self):
        frame = self.fl.load_frame()
        try:
            frame = imutils.resize(frame, width=600)
        except AttributeError:
            return
        self.out_frame = frame
        frame_number = self.fl.get_current_frame_position()
        total_frames = self.fl.get_frames_count()
        print(f"frame number: {frame_number}/{total_frames}")

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
            if frame_number % 50 == 0 or frame_number == self.fl.get_frames_count():
                self.teardown()
            self.__display.show(self.out_frame, frame_number)
            players_detected = self.players_detected[frame_number]
            players_manual = self.players_manual[frame_number]
            self.__display.show_model(players_detected, players_manual)
            self.__display.update()

    def teardown(self):
        pass

    def bootstrap(self):
        pass

    def player_pos_translated(self, pos, is_manual=True):
        frame_number = self.fl.get_current_frame_position()
        if is_manual:
            H = self.homographies_manual.get(frame_number, None)
        else:
            H = self.homographies_detected.get(frame_number, None)

        if H is not None:
            pos = self.transform_bulk_players_to_2d(pos, H)
            return int(pos[0]), int(pos[1])
        else:
            return None

    @staticmethod
    def transform_bulk_players_to_2d(players, H):
        players = np.float32(players)
        players_2d_positions = []

        for player in players:
            player = np.array(player)
            player = np.append(player, 1.)
            # https://www.learnopencv.com/homography-examples-using-opencv-python-c/
            # calculating new positions
            player_2d_position = H.dot(player)
            player_2d_position = player_2d_position / player_2d_position[2]
            players_2d_positions.append(player_2d_position)

        return players_2d_positions

    @staticmethod
    def transform_to_2d(player, H):
        player = np.float32(player)
        player = np.array(player)
        player = np.append(player, 1.)

        player_2d_position = H.dot(player)
        player_2d_position = player_2d_position / player_2d_position[2]

        return player_2d_position

    def model_to_pitchview(self, pos, is_manual=True):
        frame_number = self.fl.get_current_frame_position()
        if is_manual:
            H = self.homographies_manual.get(frame_number, None)
        else:
            H = self.homographies_detected.get(frame_number, None)

        if H is not None:
            h = inv(H)
            pos = self.transform_to_2d(pos, h)
            return int(pos[0]), int(pos[1])
        else:
            return None

    def change_player(self, player_id, new_id=None, new_color=None):
        for frame in self.players_list.players:
            for player in frame:
                if player.id == player_id:
                    if new_id is not None:
                        player.id = new_id
                    if new_color is not None:
                        player.color = new_color


if __name__ == '__main__':
    mt = PlayersPathSelector()
    FAST_ADDING = True
    mt.loop()
    mt.teardown()
