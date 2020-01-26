"""
Load player positions and preprocess them.
"""

from pitchmap.cache_loader import pickler
import numpy as np

from comparison import hausdorff_distance as mdh


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
    mdh_distance = mdh.modified_hausdorff_distance(manual_positions, detected_positions)

    print(data_file)
    print(mdh_distance)


if __name__ == '__main__':
    # data_file = "data/cache/Barca_Real_continous.mp4_path_selector_simple.pik"
    # calculate_mdh_for_file(data_file)
    data_files_ends = ["Barca_Real_continous.mp4_path_selector_middle.pik",
                       "Barca_Real_continous.mp4_path_selector_automatic.pik",
                       "Barca_Real_continous.mp4_path_selector_simple.pik",
                       "baltyk_kotwica_1.mp4_path_selector_middle.pik",
                       "baltyk_kotwica_1.mp4_path_selector_automatic.pik",
                       "baltyk_kotwica_1.mp4_path_selector_simple.pik",
                       "baltyk_starogard_1.mp4_path_selector_middle.pik",
                       "baltyk_starogard_1.mp4_path_selector_automatic.pik",
                       "baltyk_starogard_1.mp4_path_selector_simple.pik"]

    data_file_names = ["data/cache/" + file for file in data_files_ends]
    for file_name in data_file_names:
        calculate_mdh_for_file(file_name)
