import numpy as np
import cv2
from matplotlib import pyplot as plt

from pitchmap.cache_loader import pickler
from pitchmap.players import structure
import imutils
from sklearn.metrics import mean_squared_error
from scipy import interpolate


def remove_blank_spaces(x):
    indexes = []
    x_new = []
    for i, position in enumerate(x):
        if position is not None:
            indexes.append(i)
            x_new.append(position)

    indexes_all = np.array([i for i in range(len(x))])

    f2 = interpolate.interp1d(indexes, x_new, fill_value='extrapolate')
    return f2(indexes_all)


def moving_average(list, N):
    mean = [np.mean(list[x:x+N]) for x in range(len(list)-N+1)]
    return mean


def scatter_plot(frames, show_pitch=False, smooth=False):
    x = []
    y = []
    x2 = []
    y2 = []
    for f in frames.values():
        if f[0] is not None:
            x.append(f[0].position[0])
            y.append(f[0].position[1])
        if f[1] is not None:
            x2.append(f[1].position[0])
            y2.append(f[1].position[1])

    fig, ax = plt.subplots()
    pitch_model = cv2.imread('data/pitch_model.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)

    if show_pitch:
        ax.imshow(pitch_model)

    if smooth:
        x = moving_average(x, 10)
        y = moving_average(y, 10)
        x2 = moving_average(x2, 10)
        y2 = moving_average(y2, 10)

    ax.scatter(x, y, marker='s', label="wzorcowe", s=1)
    ax.scatter(x2, y2, marker='^', label="automatyczne", s=0.5)
    # ax.set_xlim([200, 300])
    # ax.set_ylim([250, 400])
    plt.legend()
    print(frames)


def positions_plot(frames, smooth=False):
    plt.style.use('ggplot')
    x = []
    y = []
    x2 = []
    y2 = []

    n = 0
    sum = 0
    for f in frames.values():
        if f[0] is not None:
            x.append(f[0].position[0])
            y.append(f[0].position[1])
        else:
            x.append(None)
            y.append(None)
        if f[1] is not None:
            x2.append(f[1].position[0])
            y2.append(f[1].position[1])
        else:
            x2.append(None)
            y2.append(None)

        if f[0] is not None and f[1] is not None:
            n += 1
            distance = np.linalg.norm(np.array([f[1].position[0], f[1].position[1]]) - np.array([f[0].position[0], f[0].position[1]]))
            sum += distance
        if n > 580:
            break

    rmse = sum/n
    manual_data = np.column_stack((x, y))
    detected_data = np.column_stack((x2, y2))
    combined_data = np.column_stack((manual_data, detected_data)).astype(float)
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]
    manual_data = combined_data[:, (0, 1)]
    detected_data = combined_data[:, (2, 3)]

    print(f"RMSE: {rmse} scikit: {mean_squared_error(manual_data, detected_data)}")


    fig, (ax, ax2) = plt.subplots(2, 1)
    pitch_model = cv2.imread('data/pitch_model.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)

    if smooth:
        x = remove_blank_spaces(x)
        x2 = remove_blank_spaces(x2)
        y = remove_blank_spaces(y)
        y2 = remove_blank_spaces(y2)

        x = moving_average(x, 15)
        y = moving_average(y, 15)
        x2 = moving_average(x2, 15)
        y2 = moving_average(y2, 15)

    ax.plot(x, label="wzorcowe")
    ax.plot(x2, label="automatyczne")
    ax.set_xlabel("Numer klatki")
    ax.set_ylabel("Pozycja x zawodnika")
    ax.legend()

    ax2.plot(y, label="wzorcowe")
    ax2.plot(y2, label="automatyczne")
    ax2.set_xlabel("Numer klatki")
    ax2.set_ylabel("Pozycja y zawodnika")
    ax2.legend()
    plt.tight_layout()
    print(frames)


if __name__ == '__main__':
    folder = "starogard"
    calibration = "simple"
    # file = f"Barca_Real_continous.mp4_path_selector_{calibration}.pik"
    # file = f"baltyk_kotwica_1.mp4_path_selector_{calibration}.pik"
    file = f"baltyk_starogard_1.mp4_path_selector_{calibration}.pik"

    data_file_name = "data/cache/" + file

    frames = pickler.unpickle_data(data_file_name)
    scatter_plot(frames, show_pitch=True, smooth=False)
    plt.savefig(f"data/images/{folder}/{calibration}.eps")
    scatter_plot(frames, show_pitch=True, smooth=True)
    plt.savefig(f"data/images/{folder}/{calibration}_smooth.eps")
    positions_plot(frames, smooth=False)
    plt.savefig(f"data/images/{folder}/{calibration}_positions.eps")
    positions_plot(frames, smooth=True)
    plt.savefig(f"data/images/{folder}/{calibration}_positions_smooth.eps")
