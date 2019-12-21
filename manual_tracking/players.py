import math
import numpy as np


class PlayersList:
    def __init__(self, frames_length):
        self.__id_counter = 0
        self.__frames_length = frames_length
        self.players = [[] for _ in range(frames_length)]

    def create_player(self, position, frame_number):
        default_id = self.default_player_id(position, frame_number)
        if default_id is None:
            default_id = self.__id_counter
            self.__id_counter += 1
        player_id = default_id

        # player_id = 60

        default_color = self.default_player_color(player_id, frame_number)
        color = int(default_color)

        return self.add_player(player_id, position, color, frame_number)

    def default_player_id(self, position, frame_number):
        if frame_number <= 0:
            return None
        nearest_player, distance = self.get_nearest_player(self.players[frame_number-1], position)
        id = None
        if distance < 20:
            if nearest_player is not None:
                id = nearest_player.id
        # check if id already exists in frame
        if id is None:
            return None
        for player in self.players[frame_number]:
            if player.id == id:
                return None
        return id

    def default_player_color(self, id, frame_number):
        if frame_number <= 0:
            return 1
        for player in self.players[frame_number-1]:
            if player.id == id:
                return player.color
        return 1

    def add_player(self, player_id, position, color, frame_number):
        p = Player(player_id, position, color, frame_number)
        self.players[frame_number].append(p)
        return p

    def find_player_id(self, position, color, frame_number):
        players_with_same_color = []
        if frame_number > 0:
            for player in self.players[frame_number-1]:
                if player.color == color:
                    players_with_same_color.append(player)

    def get_players_with_color(self, frame_number, color):
        players_with_same_color = []
        if frame_number > 0:
            for player in self.players[frame_number]:
                if player.color == color:
                    players_with_same_color.append(player)
        return players_with_same_color

    @staticmethod
    def get_nearest_player(searchable_players, position):
        position = np.array(position)
        nearest_distance = math.inf
        nearest_player = None

        for player in searchable_players:
            distance = np.linalg.norm(np.array(player.position) - position)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_player = player

        return nearest_player, nearest_distance


class Player:
    def __init__(self, player_id, position, color, frame_number):
        self.id = player_id
        self.frame_number = frame_number
        self.position = position
        self.color = color
