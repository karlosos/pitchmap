import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

from pitchmap.cache_loader import pickler
from pitchmap.homography import calibrator
from pitchmap.players import structure
from comparison import heatmap
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse

def transform_frame(frame, homo):
    """
    TODO: this was copied from pitchmap\main
    """
    rows, columns, channels = frame.shape
    columns = 600
    rows = 421

    h = self.__calibration_interactor.get_homography(frame_idx)
    return transformed_frame


def main():
    # Loading files
    input_file = "baltyk_starogard_1.mp4"
    file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypoints.pik"
    file_manual_data = f"data/cache/{input_file}_manual_tracking.pik"

    pitch_model = cv2.imread('data/pitch_model_mask.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)
    pitch_model_shape = (pitch_model.shape[1], pitch_model.shape[0])

    _, _, homographies_detected_keypoints = pickler.unpickle_data(
        file_detected_data_keypoints)
    _, homographies, _ = pickler.unpickle_data(file_manual_data)

    # Video capture
    cap = cv2.VideoCapture(f'data/{input_file}')
    print(f"Loaded video {input_file} with {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame_shape = (frame.shape[1], frame.shape[0])
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(frame_number)
        homo_pred = homographies_detected_keypoints[frame_number-1]
        homo = homographies[frame_number]

        warp = cv2.warpPerspective(frame, homo, pitch_model_shape)
        warp_pred = cv2.warpPerspective(frame, homo_pred, pitch_model_shape)

        model_warp = cv2.warpPerspective(pitch_model, np.linalg.inv(homo), frame_shape)
        model_warp_pred = cv2.warpPerspective(pitch_model, np.linalg.inv(homo_pred), frame_shape)

        # Calculate IoU
        intersection = np.logical_and(model_warp, model_warp_pred)
        union = np.logical_or(model_warp, model_warp_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU score: ", iou_score)

        cv2.imshow('orig', frame)
        cv2.imshow('warp', warp)
        cv2.imshow('warp_pred', warp_pred)
        cv2.imshow('model_warp', model_warp)
        cv2.imshow('model_warp_pred', model_warp_pred)

        k = cv2.waitKey(0)
        if k == 27:
            break

        if not ret:
            print("Can't retrieve frame")
            break







if __name__ == "__main__":
    main()