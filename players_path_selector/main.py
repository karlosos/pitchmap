from manual_tracking.frame import loader
from players_path_selector.gui import display
from players_path_selector import data_loader
from players_path_selector import transformation
from pitchmap.cache_loader import pickler

import imutils
import os


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

        self.data = data_loader.DataLoader()
        self.data.load_data(self.file_detected_data, self.file_manual_data)

        self.selected = {}
        self.id = 0

    def get_selected_player_manual(self):
        frame_number = self.fl.get_current_frame_position()
        selected_players = self.selected.get(frame_number, None)
        if selected_players is not None:
            return selected_players[0]
        return None

    def get_selected_player_detected(self):
        frame_number = self.fl.get_current_frame_position()
        selected_players = self.selected.get(frame_number, None)
        if selected_players is not None:
            return selected_players[1]
        return None

    def set_selected_player_manual(self, player):
        frame_number = self.fl.get_current_frame_position()
        selected_players = self.selected.get(frame_number, None)
        if selected_players is not None:
            self.selected[frame_number] = (player, self.selected[frame_number][1])
        else:
            self.selected[frame_number] = (player, None)

    def set_selected_player_detected(self, player):
        frame_number = self.fl.get_current_frame_position()
        selected_players = self.selected.get(frame_number, None)
        if selected_players is not None:
            self.selected[frame_number] = (self.selected[frame_number][0], player)
        else:
            self.selected[frame_number] = (None, player)

    def set_default_selected_players(self):
        frame_number = self.fl.get_current_frame_position()
        selected_players = self.selected.get(frame_number, None)
        selected_players_last_frame = self.selected.get(frame_number-1, None)

        if selected_players is not None:
            return

        if selected_players_last_frame is not None:
            player_last_frame_manual_id = selected_players_last_frame[0].id
            for player in self.data.players_manual[frame_number]:
                if player.id == player_last_frame_manual_id:
                    self.selected[frame_number] = (player, None)

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
        self.set_default_selected_players()

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
            players_detected = self.data.players_detected[frame_number]
            players_manual = self.data.players_manual[frame_number]
            self.__display.show_model(players_detected, players_manual)
            self.__display.update()

    def teardown(self):
        data_path = f"data/cache/{self.video_name}_path_selector_{self.id}.pik"
        pickler.pickle_data(self.selected,
                            data_path)
        print(f"Saved data to: {data_path}")

    def model_to_pitch_view(self, pos, is_manual=True):
        frame_number = self.fl.get_current_frame_position()
        if is_manual:
            H = self.data.homographies_manual.get(frame_number, None)
        else:
            H = self.data.homographies_detected.get(frame_number, None)

        if H is not None:
            return transformation.transform_to_3d(pos, H)
        else:
            return None


if __name__ == '__main__':
    mt = PlayersPathSelector()
    mt.id = 0
    FAST_ADDING = True
    mt.loop()
    mt.teardown()
