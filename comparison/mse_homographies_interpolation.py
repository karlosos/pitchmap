import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
from sklearn.metrics import mean_squared_error

from pitchmap.cache_loader import pickler


def calulate_iou(model_warp, model_warp_pred):
    intersection = np.logical_and(model_warp, model_warp_pred)
    union = np.logical_or(model_warp, model_warp_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def generate_points_on_image(image):
    # Generate points on warp image
    x = np.linspace(0, image.shape[1] - 1, 10)
    y = np.linspace(0, image.shape[0] - 1, 10)
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


def plot_mse_camera_angle(camera_angles, mse_scores, mse_frame_numbers=None):
    x = np.arange(len(camera_angles))
    if mse_frame_numbers is None:
        x_mse = x
    else:
        x_mse = np.array(mse_frame_numbers) - 1

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Klatka')
    ax1.set_ylabel('Kąt kamery', color=color)
    ax1.plot(x, camera_angles, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('MSE', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_mse, mse_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_compare_mse_camera_angle(camera_angles, mse_scores_list, mse_frame_numbers=None):
    x = np.arange(len(camera_angles))
    if mse_frame_numbers is None:
        x_mse = x
    else:
        x_mse = np.array(mse_frame_numbers) - 1

    fig, ax1 = plt.subplots()

    color = 'k'
    ax1.set_xlabel('Klatka')
    ax1.set_ylabel('Kąt kamery', color=color)
    ax1.plot(x, camera_angles, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('MSE', color=color)  # we already handled the x-label with ax1
    labels = ["Automatyczne", "Manualne 2 pkt", "Manualne 3 pkt"]
    for idx, mse_scores in enumerate(mse_scores_list):
        ax2.plot(x_mse[10:], mse_scores[10:], label=labels[idx])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()


def mse_for_video(camera_data, predicted_data, real_data, input_file, visualisation=False):
    pitch_model = cv2.imread('data/pitch_model.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)
    pitch_model_shape = (pitch_model.shape[1], pitch_model.shape[0])
    _, _, homographies_detected_keypoints = pickler.unpickle_data(
        predicted_data)
    _, homographies, _ = pickler.unpickle_data(real_data)
    camera_angles = pickler.unpickle_data(camera_data)
    mse_scores = []
    frame_numbers = []
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
        try:
            homo_pred = homographies_detected_keypoints[frame_number - 1]
            homo = homographies[frame_number]

            warp = cv2.warpPerspective(frame, homo, pitch_model_shape)
            warp_pred = cv2.warpPerspective(frame, homo_pred, pitch_model_shape)

            model_warp = cv2.warpPerspective(pitch_model, np.linalg.inv(homo), frame_shape)
            model_warp_pred = cv2.warpPerspective(pitch_model, np.linalg.inv(homo_pred), frame_shape)

            points = generate_points_on_image(warp)
            # Filter points
            # points = [(x, y) for (x, y) in points if not np.array_equal(warp[y, x], [0, 0, 0])]
            # Draw points
            for (x, y) in (points):
                cv2.circle(warp, (x, y), 3, (0, 0, 255), -1)

            # Transform points to frame (multiplying by inverse homography)
            frame_points = transform_points(points, homo, inverse=True)
            # Draw on frame
            for (x, y, _) in frame_points:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            pred_points = transform_points(frame_points, homo_pred, inverse=False)
            for (x, y, _) in pred_points:
                cv2.circle(warp_pred, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Calculate MSE
            mse = mean_squared_error(points, np.array(pred_points)[:, :2])
            mse_scores.append(mse)
            frame_numbers.append(frame_number)

            # Visualisation
            if visualisation:
                model = pitch_model.copy()
                for (x, y) in (points):
                    cv2.circle(model, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow('model', model)
                pred_model = pitch_model.copy()
                for (x, y, _) in pred_points:
                    cv2.circle(pred_model, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.imshow('orig', frame)
                cv2.imshow('pred_model', pred_model)
                # cv2.imshow('warp', warp)
                # cv2.imshow('warp_pred', warp_pred)
                # cv2.imshow('model_warp', model_warp)
                # cv2.imshow('model_warp_pred', model_warp_pred)

                k = cv2.waitKey(0)
                if k == 27:
                    break
        except:
            pass
            # print("Could not load homography")
    print("Average MSE for video sequence:", np.mean(mse_scores))

    return camera_angles, mse_scores, frame_numbers


def main():
    # Loading files
    input_file = "Baltyk_Koszalin_05_06.mp4"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypoints.pik"
    file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"
    file_manual_data = f"data/cache/{input_file}_manual_tracking.pik"
    file_camera_movement = f"data/cache/{input_file}_CameraMovementAnalyser.pik"
    visualisation = True

    camera_angles, mse_scores, mse_frame_numbers = mse_for_video(file_camera_movement, file_detected_data_keypoints,
                                                                 file_manual_data,
                                                                 input_file, visualisation)

    plot_mse_camera_angle(camera_angles, mse_scores, mse_frame_numbers)


if __name__ == "__main__":
    main()
