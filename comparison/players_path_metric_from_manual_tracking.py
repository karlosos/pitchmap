"""
Load player positions and preprocess them.
"""

from pitchmap.cache_loader import pickler
import numpy as np
import matplotlib.pyplot as plt

from comparison import hausdorff_distance as mdh
from scipy import interpolate

from path_plot import pitch_plot


def remove_blank_spaces(x):
    indexes = []
    x_new = []
    for i, position in enumerate(x):
        if position is not None:
            indexes.append(i)
            x_new.append(position)

    indexes_all = np.array([i for i in range(len(x))])

    f2 = interpolate.interp1d(indexes, x_new, fill_value='extrapolate')
    return f2(indexes_all)


def moving_average(list, N):
    mean = [np.mean(list[x:x + N]) for x in range(len(list) - N + 1)]
    return mean


def post_processing(data):
    new_x = remove_blank_spaces(data[:, 0])
    new_y = remove_blank_spaces(data[:, 1])

    new_x = moving_average(new_x, 15)
    new_y = moving_average(new_y, 15)
    return np.column_stack((new_x, new_y))


def calculate_mdh_for_file(data_file):
    players_manual, homographies_manual, _ = pickler.unpickle_data(data_file + "_manual_tracking.pik")
    players_manual = players_manual.players
    _, _, homographies_keypoints = pickler.unpickle_data(data_file + "_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik")
    _, _, homographies_2points = pickler.unpickle_data(data_file + "_PlayersListComplex_CalibrationInteractorAutomatic.pik")
    _, _, homographies_3points = pickler.unpickle_data(data_file + "_PlayersListComplex_CalibrationInteractorMiddlePoint.pik")

    positions_manual = []
    positions_keypoints = []
    positions_2points = []
    positions_3points = []

    # Get manual positions
    for idx in range(len(players_manual)):
        homo_manual = homographies_manual.get(idx)
        if len(players_manual[idx]) >= 1:
            player_positions = players_manual[idx][0].position
            positions_manual.append(player_positions)

            if homo_manual is not None:
                player_positions_frame = transform_positions(player_positions, np.linalg.inv(homo_manual))
                transform_positions_for_frame(homographies_keypoints, idx, player_positions_frame, positions_keypoints)
                transform_positions_for_frame(homographies_2points, idx, player_positions_frame, positions_2points)
                transform_positions_for_frame(homographies_3points, idx, player_positions_frame, positions_3points)

    positions_manual = np.array(positions_manual)
    positions_keypoints = np.array(positions_keypoints)
    positions_2points = np.array(positions_2points)
    positions_3points = np.array(positions_3points)

    # Plot pitch model with positions
    pitch_plot([positions_manual, positions_keypoints, positions_2points, positions_3points], show_pitch=True, smooth=True)
    plt.show()

    if True:
        positions_manual = post_processing(positions_manual)
        positions_keypoints = post_processing(positions_keypoints)
        positions_2points = post_processing(positions_2points)
        positions_3points = post_processing(positions_3points)

    mdh_distance_keypoints = mdh.modified_hausdorff_distance(positions_manual, positions_keypoints)
    mdh_distance_2points = mdh.modified_hausdorff_distance(positions_manual, positions_2points)
    mdh_distance_3points = mdh.modified_hausdorff_distance(positions_manual, positions_3points)

    print(data_file, mdh_distance_keypoints, mdh_distance_2points, mdh_distance_3points)

    return mdh_distance_keypoints, mdh_distance_2points, mdh_distance_3points


def transform_positions_for_frame(homographies_detected, idx, player_positions_frame, positions_detected):
    homo_keypoints = safe_list_get(homographies_detected, idx)
    if homo_keypoints is not None:
        positions_detected.append(transform_positions(player_positions_frame, homo_keypoints))
    # else:
    #     positions_detected.append(None)


def safe_list_get (l, idx, default=None):
    try:
        return l[idx]
    except IndexError:
        return default


def transform_positions(player, H):
    player = np.float32(player)
    player = np.array(player)
    if player.shape[0] == 2:
        player = np.append(player, 1.)

    player_2d_position = H.dot(player)
    player_2d_position = player_2d_position / player_2d_position[2]

    return player_2d_position


if __name__ == '__main__':
    video_path = [
        "baltyk_koszalin_02.mp4",
        "baltyk_koszalin_03_03.mp4",
        "baltyk_koszalin_04_04.mp4",
        "baltyk_koszalin_05_06.mp4",
        "baltyk_koszalin_06_07.mp4",
        "baltyk_koszalin_07_09.mp4",
        "baltyk_kotwica_1.mp4",
        "baltyk_starogard_1.mp4",
        "WDA_Kotwica_01.mp4",
        "ENG_POL_01_09.mp4",
        "BAR_SEV_01.mp4",
    ]

    data_file_names = ["data/cache/" + file for file in video_path]
    for file_name in data_file_names:
        calculate_mdh_for_file(file_name)
        # mdh_distance_keypoints, mdh_distance_2points, mdh_distance_3points = calculate_mdh_for_file(file_name)
