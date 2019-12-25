import numpy as np
from numpy.linalg import inv


def bulk_transform_to_2d(players, H):
    players = np.float32(players)
    players_2d_positions = []

    for player in players:
        player = np.array(player)
        player = np.append(player, 1.)
        player_2d_position = H.dot(player)
        player_2d_position = player_2d_position / player_2d_position[2]
        players_2d_positions.append(player_2d_position)

    return players_2d_positions


def transform_to_2d(player, H):
    player = np.float32(player)
    player = np.array(player)
    player = np.append(player, 1.)

    player_2d_position = H.dot(player)
    player_2d_position = player_2d_position / player_2d_position[2]

    return player_2d_position


def transform_to_3d(pos, H):
    h = inv(H)
    pos = transform_to_2d(pos, h)
    return int(pos[0]), int(pos[1])
