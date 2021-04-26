import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pitchmap.cache_loader import pickler


def distance_metric(points_a, points_b):
    num_points = points_a.shape[0]
    distances = []
    for i in range(num_points):
        dist = distance.euclidean(points_a[i], points_b[i])
        distances.append(dist)
    return np.mean(distances) * 0.2  # mnożenie przez 0.2 aby mieć wynik w metrach. bo 80 pikseli to 16 metrów.


def update_mean_distance_map(mean_distance_map, real_points, predicted_points):
    num_points = real_points.shape[0]
    for i in range(num_points):
        (x, y) = real_points[i]
        dist = distance.euclidean(real_points[i], predicted_points[i])
        mean_distance_map[(x, y)] += np.array([dist * 0.2, 1])  # dodanie do słownika odległości w metrach i licznika
    return mean_distance_map


def generate_points_on_image(image, spacing=None):
    # Generate points on warp image
    x = np.linspace(0, image.shape[1] - 1, 10)
    y = np.linspace(0, image.shape[0] - 1, 10)
    if spacing is not None:
        x = np.linspace(0, image.shape[1] - 1, spacing)
        y = np.linspace(0, image.shape[0] - 1, spacing)
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


def plot_compare_mean_distance_camera_angle(camera_angles, mean_distance_scores_list, mean_distance_frame_numbers=None):
    x = np.arange(len(camera_angles))
    if mean_distance_frame_numbers is None:
        x_md = x
    else:
        x_md = np.array(mean_distance_frame_numbers) - 1

    fig, ax1 = plt.subplots()

    color = 'k'
    ax1.set_xlabel('Klatka')
    ax1.set_ylabel('Wychylenie kamery', color=color)
    ax1.plot(x, camera_angles, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Średni błąd [m]', color=color)  # we already handled the x-label with ax1
    labels = ["Automatyczne", "Manualne 2 pkt", "Manualne 3 pkt"]
    for idx, md_scores in enumerate(mean_distance_scores_list):
        ax2.plot(x_md[10:], md_scores[10:], label=labels[idx])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()


def mean_distance_for_video(camera_data, predicted_data, real_data, input_file, visualisation=False):
    pitch_model = cv2.imread('data/pitch_model.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)
    pitch_model_shape = (pitch_model.shape[1], pitch_model.shape[0])
    _, _, homographies_detected_keypoints = pickler.unpickle_data(
        predicted_data)
    _, homographies, _ = pickler.unpickle_data(real_data)
    camera_angles = pickler.unpickle_data(camera_data)
    mean_distance_scores = []
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
            points = [(x, y) for (x, y) in points if not np.array_equal(warp[y, x], [0, 0, 0])]
            points = np.array(points)
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
            # mse = mean_squared_error(points, np.array(pred_points)[:, :2])
            mean_distance = distance_metric(points, np.array(pred_points)[:, :2])
            mean_distance_scores.append(mean_distance)
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
        except Exception as e:
            print(e)
            # print("Could not load homography")
            pass
    print("Average MSE for video sequence:", np.mean(mean_distance_scores))

    mean_distance_map = {pos: dst / cnt for (pos, (dst, cnt)) in mean_distance_map.items()}
    print(mean_distance_map)
    return camera_angles, mean_distance_scores, frame_numbers


def map_for_input_video(predicted_data, real_data, input_file):
    pitch_model = load_pitch_model()
    pitch_model_shape = (pitch_model.shape[1], pitch_model.shape[0])
    _, _, homographies_detected_keypoints = pickler.unpickle_data(
        predicted_data)
    _, homographies, _ = pickler.unpickle_data(real_data)
    # Video capture
    cap = cv2.VideoCapture(f'data/{input_file}')
    print(f"Loaded video {input_file} with {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")

    mean_distance_map = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't retrieve frame")
            break

        frame = imutils.resize(frame, width=600)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        try:
            homo_pred = homographies_detected_keypoints[frame_number - 1]
            homo = homographies[frame_number]
            warp = cv2.warpPerspective(frame, homo, pitch_model_shape)
            points = generate_points_on_image(warp, spacing=100)
            if mean_distance_map is None:
                mean_distance_map = {(x, y): np.array([0.0, 0.0]) for (x, y) in points}
            # Filter points
            points = [(x, y) for (x, y) in points if not np.array_equal(warp[y, x], [0, 0, 0])]
            points = np.array(points)
            # Transform points to frame (multiplying by inverse homography)
            frame_points = transform_points(points, homo, inverse=True)
            pred_points = transform_points(frame_points, homo_pred, inverse=False)
            # Prepare pitch map with errors
            mean_distance_map = update_mean_distance_map(mean_distance_map, points, np.array(pred_points)[:, :2])
        except Exception as e:
            # print(e)
            pass
    return mean_distance_map


def load_pitch_model():
    pitch_model = cv2.imread('data/pitch_model.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)
    return pitch_model


def generate_map_image(mean_distance_map, title=None):
    pitch_model = load_pitch_model()
    # Normalising map
    mean_distance_map = {pos: dst / cnt for (pos, (dst, cnt)) in mean_distance_map.items()}
    # Generating image with map
    grid_x, grid_y = np.mgrid[0:pitch_model.shape[1] - 1, 0:pitch_model.shape[0] - 1]
    points = np.array(list(mean_distance_map.keys()))
    values = np.array(list(mean_distance_map.values()))
    grid = griddata(points, values, (grid_x, grid_y), method='linear')
    fig, ax = plt.subplots(1)
    ax.imshow(pitch_model)
    pitch_im = ax.imshow(grid.T, extent=(0, pitch_model.shape[1] - 1, 0, pitch_model.shape[0] - 1), origin='upper', alpha=.9)
    pitch_im.set_clim(0, 10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    ax.axis('off')
    cbar = fig.colorbar(pitch_im, cax=cax)
    cbar.minorticks_on()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()



def main_metric():
    # Loading files
    input_file = "Baltyk_Koszalin_02.mp4"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypoints.pik"
    file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"
    file_manual_data = f"data/cache/{input_file}_manual_tracking.pik"
    file_camera_movement = f"data/cache/{input_file}_CameraMovementAnalyser.pik"
    visualisation = False

    camera_angles, mean_distance_scores, mean_distance_frame_numbers = mean_distance_for_video(file_camera_movement,
                                                                                               file_detected_data_keypoints,
                                                                                               file_manual_data,
                                                                                               input_file,
                                                                                               visualisation)

    plot_compare_mean_distance_camera_angle(camera_angles, [mean_distance_scores], mean_distance_frame_numbers)
    plt.show()


def main_map():
    # Loading files
    input_file = "Baltyk_Koszalin_02.mp4"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypoints.pik"
    file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorKeypointsAdvanced.pik"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorAutomatic.pik"
    # file_detected_data_keypoints = f"data/cache/{input_file}_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"
    file_manual_data = f"data/cache/{input_file}_manual_tracking.pik"

    mean_distance_map = map_for_input_video(file_detected_data_keypoints, file_manual_data, input_file)
    generate_map_image(mean_distance_map)
    plt.show()


if __name__ == "__main__":
    # main_metric()
    main_map()
