from iou_homographies_interpolation import iou_for_video
import numpy as np


def compare_methods(input_file):
    keypoints_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_2points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    manual_3points_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"

    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"

    keypoints_iou_scores = iou_for_video(keypoints_data_file, manual_data_file, input_file)
    manual_2points_iou_scores = iou_for_video(manual_2points_data_file, manual_data_file, input_file)
    manual_3points_iou_scores = iou_for_video(manual_3points_data_file, manual_data_file, input_file)

    print("Keypoints:", np.mean(keypoints_iou_scores), np.median(keypoints_iou_scores))
    print("Manual 2 points:", np.mean(manual_2points_iou_scores), np.median(manual_2points_iou_scores))
    print("Manual 3 points:", np.mean(manual_3points_iou_scores), np.median(manual_3points_iou_scores))


def single_file():
    # Loading files
    input_file = "baltyk_kotwica_1.mp4"
    predicted_data_file = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    manual_data_file = f"data/cache/{input_file}_manual_tracking.pik"
    visualisation_flag = False

    iou_scores = iou_for_video(predicted_data_file, manual_data_file, input_file, visualisation_flag)

    print(predicted_data_file)
    print("Average IOU for video sequence:", np.mean(iou_scores))


if __name__ == '__main__':
    # single_file()
    compare_methods(input_file="baltyk_kotwica_1.mp4")