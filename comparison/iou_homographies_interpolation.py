import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

from pitchmap.cache_loader import pickler


def calulate_iou(model_warp, model_warp_pred):
    intersection = np.logical_and(model_warp, model_warp_pred)
    union = np.logical_or(model_warp, model_warp_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def main():
    # Loading files
    input_file = "BAR_SEV_01.mp4"
    file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    file_manual_data = f"data/cache/{input_file}_manual_tracking.pik"

    pitch_model = cv2.imread('data/pitch_model_mask.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)
    pitch_model_shape = (pitch_model.shape[1], pitch_model.shape[0])

    pitch_model_border = cv2.imread('data/pitch_model_border.png')
    pitch_model_border = imutils.resize(pitch_model_border, width=600)
    pitch_model_border = cv2.bitwise_not(pitch_model_border)

    _, _, homographies_detected_keypoints = pickler.unpickle_data(
        file_detected_data_keypoints)
    _, homographies, _ = pickler.unpickle_data(file_manual_data)
    iou_scores = []

    # Video capture
    cap = cv2.VideoCapture(f'data/{input_file}')
    print(f"Loaded video {input_file} with {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't retrieve frame")
            break

        frame = imutils.resize(frame, width=600)
        frame_shape = (frame.shape[1], frame.shape[0])
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(frame_number)
        try:
            homo_pred = homographies_detected_keypoints[frame_number-1]
            homo = homographies[frame_number]

            warp = cv2.warpPerspective(frame, homo, pitch_model_shape)
            warp_pred = cv2.warpPerspective(frame, homo_pred, pitch_model_shape)

            model_warp = cv2.warpPerspective(pitch_model, np.linalg.inv(homo), frame_shape)
            model_warp_pred = cv2.warpPerspective(pitch_model, np.linalg.inv(homo_pred), frame_shape)

            model_border_warp = cv2.warpPerspective(pitch_model_border, np.linalg.inv(homo), frame_shape)
            model_border_warp_pred = cv2.warpPerspective(pitch_model_border, np.linalg.inv(homo_pred), frame_shape)

            # Calculate IoU
            iou_score = calulate_iou(model_warp, model_warp_pred)
            iou_scores.append(iou_score)
            print("IoU score: ", iou_score)

            # Visualisation
            cv2.imshow('orig', frame)
            cv2.imshow('warp', warp)
            cv2.imshow('warp_pred', warp_pred)
            cv2.imshow('model_warp', model_warp)
            cv2.imshow('model_warp_pred', model_warp_pred)

            k = cv2.waitKey(0)
            if k == 27:
                break
        except Exception:
            print("Could not load homography")


    print("Average IOU for video sequence:", np.mean(iou_scores))


if __name__ == "__main__":
    main()