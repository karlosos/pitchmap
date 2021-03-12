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


def generate_points_on_image(image):
    # Generate points on warp image
    x = np.linspace(0, image.shape[1]-1, 20)
    y = np.linspace(0, image.shape[0]-1, 20)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()]).astype(int)

    return points


def transform_points(points, homo, inverse):
    points = np.float32(points)
    warped_positions = []
    for circle in points.tolist():
        circle = np.array(circle)
        if circle.size == 2:
            circle = np.append(circle, 1.)
        if inverse:
            circle_warped = np.linalg.inv(homo).dot(circle)
        else:
            circle_warped = homo.dot(circle)
        circle_warped = circle_warped / circle_warped[2]
        warped_positions.append(circle_warped)
    return warped_positions


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
        homo_pred = homographies_detected_keypoints[frame_number-1]
        homo = homographies[frame_number]

        warp = cv2.warpPerspective(frame, homo, pitch_model_shape)
        warp_pred = cv2.warpPerspective(frame, homo_pred, pitch_model_shape)

        model_warp = cv2.warpPerspective(pitch_model, np.linalg.inv(homo), frame_shape)
        model_warp_pred = cv2.warpPerspective(pitch_model, np.linalg.inv(homo_pred), frame_shape)

        points = generate_points_on_image(warp)
        # Filter points
        points = [(x, y) for (x, y) in points if not np.array_equal(warp[y, x], [0, 0, 0])]
        # Draw points
        for (x, y) in (points):
            cv2.circle(warp, (x, y), 2, (0, 0, 255), -1)

        # Transform points to frame (multiplying by inverse homography)
        frame_points = transform_points(points, homo, inverse=True)
        # Draw on frame
        for (x, y, _) in frame_points:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

        pred_points = transform_points(frame_points, homo_pred, inverse=False)
        for (x, y, _) in pred_points:
            cv2.circle(warp_pred, (int(x), int(y)), 2, (255, 0, 0), -1)

        # Visualisation
        cv2.imshow('orig', frame)
        cv2.imshow('warp', warp)
        cv2.imshow('warp_pred', warp_pred)
        # cv2.imshow('model_warp', model_warp)
        # cv2.imshow('model_warp_pred', model_warp_pred)
        #
        k = cv2.waitKey(0)
        if k == 27:
            break

    print("Average IOU for video sequence:", np.mean(iou_scores))


if __name__ == "__main__":
    main()
