from iou_homographies_interpolation import iou_for_video
import numpy as np
import pandas as pd


def compare_methods(input_file):
    keypoints_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_2points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    manual_3points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"

    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"

    keypoints_iou_scores = iou_for_video(keypoints_data_file, manual_data_file, input_file)
    manual_2points_iou_scores = iou_for_video(manual_2points_data_file, manual_data_file, input_file)
    manual_3points_iou_scores = iou_for_video(manual_3points_data_file, manual_data_file, input_file)

    print("=============")
    print(input_file)
    print("=============")
    k_mean = np.mean(keypoints_iou_scores)
    k_median = np.median(keypoints_iou_scores)
    print("Keypoints:", k_mean, k_median)
    m_2_mean = np.mean(manual_2points_iou_scores)
    m_2_median = np.median(manual_2points_iou_scores)
    print("Manual 2 points:", m_2_mean, m_2_median)
    m_3_mean = np.mean(manual_3points_iou_scores)
    m_3_median = np.median(manual_3points_iou_scores)
    print("Manual 3 points:", m_3_mean, m_3_median)

    return k_mean, k_median, m_2_mean, m_2_median, m_3_mean, m_3_median


def single_file():
    # Loading files
    input_file = "baltyk_kotwica_1.mp4"
    predicted_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"
    visualisation_flag = False

    iou_scores = iou_for_video(predicted_data_file, manual_data_file, input_file, visualisation_flag)

    print(predicted_data_file)
    print("Average IOU for video sequence:", np.mean(iou_scores))


def batch_files():
    """
    Do masowego sprawdzania iou
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

    data = {"file": [], "k_mean": [], "k_median": [], "m_2_mean": [], "m_2_median": [], "m_3_mean": [], "m_3_median": []}

    for file in files:
        k_mean, k_median, m_2_mean, m_2_median, m_3_mean, m_3_median = compare_methods(input_file=file)
        data["file"].append(file)
        data["k_mean"].append(k_mean)
        data["k_median"].append(k_median)
        data["m_2_mean"].append(m_2_mean)
        data["m_2_median"].append(m_2_median)
        data["m_3_mean"].append(m_3_mean)
        data["m_3_median"].append(m_3_median)

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("data/cache/batch_iou_homographies.csv")


def load_data():
    """
    Do wczytywania wcześniej wyznaczonych danych
    """
    df = pd.read_csv("data/cache/batch_iou_homographies.csv")
    print(df.describe())


if __name__ == '__main__':
    # single_file()
    # batch_files()
    load_data()

