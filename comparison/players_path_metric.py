"""
Load player positions and preprocess them.
"""

from pitchmap.cache_loader import pickler
import numpy as np

from comparison import hausdorff_distance as mdh
from scipy import interpolate


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
    frames = pickler.unpickle_data(data_file)
    manual_positions = []
    detected_positions = []

    for f in frames.values():
        if f[0] is not None:
            manual_positions.append(f[0].position)
        if f[1] is not None:
            detected_positions.append(f[1].position)

    manual_positions = np.array(manual_positions)
    detected_positions = np.array(detected_positions)

    if True:
        manual_positions = post_processing(manual_positions)
        detected_positions = post_processing(detected_positions)

    mdh_distance = mdh.modified_hausdorff_distance(manual_positions, detected_positions)

    print(data_file)
    print(mdh_distance)


if __name__ == '__main__':
    # data_file = "data/cache/Barca_Real_continous.mp4_path_selector_simple.pik"
    # calculate_mdh_for_file(data_file)
    data_files_ends = [
        "Barca_Real_continous.mp4_path_selector_middle.pik",
        "Barca_Real_continous.mp4_path_selector_automatic.pik",
        "Barca_Real_continous.mp4_path_selector_simple.pik",
        "baltyk_kotwica_1.mp4_path_selector_middle.pik",
        "baltyk_kotwica_1.mp4_path_selector_automatic.pik",
        "baltyk_kotwica_1.mp4_path_selector_simple.pik",
        "baltyk_starogard_1.mp4_path_selector_middle.pik",
        "baltyk_starogard_1.mp4_path_selector_automatic.pik",
        "baltyk_starogard_1.mp4_path_selector_simple.pik",
    ]

    data_file_names = ["data/cache/" + file for file in data_files_ends]
    for file_name in data_file_names:
        calculate_mdh_for_file(file_name)
