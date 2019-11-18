"""
player.py is module with data structures for storing player positions and their colors (acutally team ids).
After scanning every frame for footballers they're stored in these structures.
Positions in these structures are related to position on video not absolute position on pitch.
Player = footballer
"""


import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class PlayerList(metaclass=ABCMeta):
    @abstractmethod
    def get_players_positions_from_frame(self, frame_number):
        pass

    @abstractmethod
    def get_players_team_ids_from_frame(self, frame_number):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def find_player_id(self, position, color, frame_number):
        pass

    @abstractmethod
    def add_player(self, player_id, position, color, frame_number):
        pass

    @abstractmethod
    def assign_player(self, position, color, frame_number):
        pass

    @abstractmethod
    def clone_player(self, player, position, color, frame_number):
        pass


class PlayersListSimple(PlayerList):
    def __init__(self, frames_length):
        self.players = []
        self.colors = []
        self.__frames_length = frames_length

    def clear(self):
        self.players = []
        self.colors = []

    def get_players_positions_from_frame(self, frame_number):
        print(f"Positions: {self.players}")
        return self.players

    def get_players_team_ids_from_frame(self, frame_number):
        print(f"Colors: {self.colors}")
        return self.colors

    def assign_player(self, position, color, frame_number):
        self.players.append(position)
        self.colors.append(color)
        return PlayerSimple(position, color)

    def find_player_id(self, position, color, frame_number):
        pass

    def add_player(self, player_id, position, color, frame_number):
        pass

    def clone_player(self, player, position, color, frame_number):
        pass


class PlayersListComplex(PlayerList):
    def __init__(self, frames_length):
        self.__id_counter = 0
        self.__frames_length = frames_length
        self.players = [{} for _ in range(frames_length + 1)]

    def clear(self):
        pass

    def get_players_positions_from_frame(self, frame_number):
        players = self.players[frame_number] if frame_number < self.__frames_length else []
        if players:
            if type(players) is dict:
                players = list(players.values())
            positions = [player.position for player in players]
        else:
            positions = []

        return positions

    def get_players_team_ids_from_frame(self, frame_number):
        #print(f"get colors from frame {frame_number}")
        players = self.players[frame_number] if frame_number < self.__frames_length else []
        if players:
            if type(players) is dict:
                players = list(players.values())
            colors = list(map(lambda player: player.calculate_real_color(), players))
        else:
            colors = []
        #print(f"colors: {colors}")
        return colors

    def assign_player(self, position, color, frame_number):
        player_id = self.find_player_id(position, color, frame_number)
        player = self.add_player(player_id=player_id, position=position, color=color, frame_number=frame_number)
        return player

    def find_player_id(self, position, color, frame_number):
        if frame_number-1 >= 0:
            players_last_frame = copy.copy(self.players[frame_number-1])

            # removing existing id's
            keys_to_remove = self.players[frame_number].keys()
            for k in keys_to_remove:
                players_last_frame.pop(k, None)

            closer_position = {key: player for key, player in players_last_frame.items() if
                               np.linalg.norm(np.asarray(player.position) - np.asarray(position)) < 10}
            same_color = {key: player for key, player in closer_position.items() if player.color == color}
            if len(same_color) == 1:
                return next(iter(same_color.values())).id  # return first value
            elif len(closer_position) == 1:
                return next(iter(closer_position.values())).id  # return only value

            if len(closer_position) > 1:
                closest_distance = 10+1
                closest_id = None
                for key, player in closer_position.items():
                    distance = np.linalg.norm(np.asarray(player.position) - np.asarray(position))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_id = key
                return closest_id
        return None

    def add_player(self, player_id, position, color, frame_number):
        if player_id is None:
            player_id = self.__id_counter
            self.__id_counter += 1

        p = PlayerComplex(self, player_id, position, color, frame_number)
        self.players[frame_number][player_id] = p
        return p

    def clone_player(self, player, position, color, frame_number):
        p = PlayerComplex(self, player.id, position, color, frame_number)
        self.players[frame_number][player.id] = p
        return p

    def is_frame_populated(self, frame_number):
        if self.players[frame_number]:
            return True
        else:
            return False


class PlayerSimple:
    def __init__(self, position, color):
        self.position = position
        self.color = color
        self.id = 0
        self.__frame_number = 0

    def calculate_real_color(self):
        return self.color


class PlayerComplex:
    def __init__(self, player_list, player_id, position, color, frame_number):
        self.__player_list = player_list
        self.id = player_id
        self.__frame_number = frame_number
        self.position = position
        self.color = color

    def calculate_real_color(self):
        left_bound = self.__frame_number - 10 if self.__frame_number - 10 > 0 else 0
        last_colors = []
        for i in range(left_bound, self.__frame_number+1):
            p = self.__player_list.players[i].get(self.id)
            if p is not None:
                color = p.color
            else:
                color = -1
            last_colors.append(color)

        #print(f"Player: {self.id} last colors: {last_colors} for frame {self.__frame_number}")
        last_colors = np.array(last_colors)
        last_colors = last_colors[last_colors != -1]
        real_color = np.bincount(last_colors).argmax()
        return real_color
