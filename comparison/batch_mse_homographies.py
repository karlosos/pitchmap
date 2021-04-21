from mse_homographies_interpolation import mse_for_video, plot_compare_mse_camera_angle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_methods(input_file):
    keypoints_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_2points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    manual_3points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"

    camera_movement_file = f"data/cache/{input_file}_CameraMovementAnalyser.pik"
    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"

    camera_angles, keypoints_mse_scores, mse_frame_numbers = mse_for_video(camera_movement_file, keypoints_data_file,
                                                                           manual_data_file, input_file)
    _, manual_2points_mse_scores, _ = mse_for_video(camera_movement_file, manual_2points_data_file, manual_data_file,
                                                    input_file)
    _, manual_3points_mse_scores, _ = mse_for_video(camera_movement_file, manual_3points_data_file, manual_data_file,
                                                    input_file)

    plot_compare_mse_camera_angle(camera_angles,
                                  [keypoints_mse_scores, manual_2points_mse_scores, manual_3points_mse_scores],
                                  mse_frame_numbers)
    plt.savefig(f"data/experiments/mse/{input_file}.eps")
    # plt.show()

    print("=============")
    print(input_file)
    print("=============")
    k_mean = np.mean(keypoints_mse_scores[2:])  # 120: for WDA_Kotwica_01, 2: because first frame has big error
    k_median = np.median(keypoints_mse_scores[2:])
    print("Keypoints:", k_mean, k_median)
    m_2_mean = np.mean(manual_2points_mse_scores[2:])
    m_2_median = np.median(manual_2points_mse_scores[2:])
    print("Manual 2 points:", m_2_mean, m_2_median)
    m_3_mean = np.mean(manual_3points_mse_scores[2:])
    m_3_median = np.median(manual_3points_mse_scores[2:])
    print("Manual 3 points:", m_3_mean, m_3_median)
    print("Compared frames:", len(keypoints_mse_scores))

    return k_mean, k_median, m_2_mean, m_2_median, m_3_mean, m_3_median, len(keypoints_mse_scores)


def single_file():
    # Loading files
    input_file = "baltyk_kotwica_1.mp4"
    predicted_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"
    camera_movement_file = f"data/cache/{input_file}_CameraMovementAnalyser.pik"

    camera_angles, mse_scores, mse_frame_numbers = mse_for_video(camera_movement_file, predicted_data_file,
                                                                 manual_data_file, input_file)

    print(predicted_data_file)
    print("Average IOU for video sequence:", np.mean(mse_scores))

    plot_compare_mse_camera_angle(camera_angles, [mse_scores], mse_frame_numbers)


def batch_files():
    """
    Do masowego sprawdzania mse
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
        "WDA_Kotwica_01.mp4",
        "ENG_POL_01_09.mp4",
        "BAR_SEV_01.mp4",
    ]

    data = {"file": [], "k_mean": [], "k_median": [], "m_2_mean": [], "m_2_median": [], "m_3_mean": [],
            "m_3_median": [], "frames": []}

    for file in files:
        k_mean, k_median, m_2_mean, m_2_median, m_3_mean, m_3_median, num_frames = compare_methods(input_file=file)
        data["file"].append(file)
        data["k_mean"].append(k_mean)
        data["k_median"].append(k_median)
        data["m_2_mean"].append(m_2_mean)
        data["m_2_median"].append(m_2_median)
        data["m_3_mean"].append(m_3_mean)
        data["m_3_median"].append(m_3_median)
        data["frames"].append(num_frames)

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("data/experiments/batch_mse_homographies.csv")


def load_data():
    """
    Do wczytywania wcześniej wyznaczonych danych
    """
    df = pd.read_csv("data/experiments/batch_mse_homographies.csv")
    print(df.describe())


if __name__ == '__main__':
    # single_file()
    compare_methods("WDA_Kotwica_01.mp4")
    # batch_files()
    # load_data()
