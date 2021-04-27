from mean_distance_homographies_interpolation import map_for_input_video, generate_map_image
import matplotlib.pyplot as plt
import numpy as np


def compare_methods(input_file):
    keypoints_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_2points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    manual_3points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"

    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"

    keypoints_map = map_for_input_video(keypoints_data_file, manual_data_file, input_file)
    manual_2points_map = map_for_input_video(manual_2points_data_file, manual_data_file, input_file)
    manual_3points_map = map_for_input_video(manual_3points_data_file, manual_data_file, input_file)

    return keypoints_map, manual_2points_map, manual_3points_map
    # return keypoints_map, None, None


def batch_files():
    """
    Do masowego sprawdzania mean distance
    """
    files = [
        "baltyk_koszalin_02.mp4",
        "baltyk_koszalin_03_03.mp4",
        "baltyk_koszalin_04_04.mp4",
        "baltyk_koszalin_05_06.mp4",
        "baltyk_koszalin_06_07.mp4",
        "baltyk_koszalin_07_09.mp4",
        "baltyk_kotwica_1.mp4",
        "baltyk_starogard_1.mp4",
        # "WDA_Kotwica_01.mp4",
        "ENG_POL_01_09.mp4",
        "BAR_SEV_01.mp4",
    ]

    keypoints_combined_map = None
    manual_2points_combined_map = None
    manual_3points_combined_map = None

    for file in files:
        keypoints_map, manual_2points_map, manual_3points_map = compare_methods(input_file=file)
        generate_map_image(keypoints_map, title=file)
        plt.savefig(f"data/experiments/maps/{file}_keypoints.pdf")

        generate_map_image(manual_2points_map, title=file)
        plt.savefig(f"data/experiments/maps/{file}_2points.pdf")

        generate_map_image(manual_3points_map, title=file)
        plt.savefig(f"data/experiments/maps/{file}_3points.pdf")
        # plt.show()

        # Combine map
        keypoints_combined_map = combine_map(keypoints_combined_map, keypoints_map)
        manual_2points_combined_map = combine_map(manual_2points_combined_map, manual_2points_map)
        manual_3points_combined_map = combine_map(manual_3points_combined_map, manual_3points_map)

    generate_map_image(keypoints_combined_map)
    plt.savefig(f"data/experiments/maps/keypoints_combined_map.pdf")
    plt.show()

    generate_map_image(manual_2points_combined_map)
    plt.savefig(f"data/experiments/maps/2points_combined_map.pdf")
    plt.show()

    generate_map_image(manual_3points_combined_map)
    plt.savefig(f"data/experiments/maps/3points_combined_map.pdf")
    plt.show()


def combine_map(combined_map, map):
    if combined_map is None:
        combined_map = map
    else:
        combined_map = {pos: value + map[pos] for (pos, value) in combined_map.items()}
    return combined_map


if __name__ == '__main__':
    batch_files()
