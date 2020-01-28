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

    # spline
    s = interpolate.InterpolatedUnivariateSpline(indexes, x_new)
    indexes_all = np.array([i for i in range(len(x))])
    x_interpolated = s(indexes_all)
    # return x_interpolated

    # linear
    f2 = interpolate.interp1d(indexes, x_new)
    return f2(indexes_all)


def moving_average(list, N):
    mean = [np.mean(list[x:x+N]) for x in range(len(list)-N+1)]
    return mean


def scatter_plot(frames, show_pitch=False):
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

    # Filtracja
    x = moving_average(x, 10)
    y = moving_average(y, 10)
    x2 = moving_average(x2, 10)
    y2 = moving_average(y2, 10)

    ax.scatter(x, y, marker='s', label="manual", s=1)
    ax.scatter(x2, y2, marker='^', label="automatic", s=0.5)
    # ax.set_xlim([200, 300])
    # ax.set_ylim([250, 400])
    plt.legend()
    plt.show()
    print(frames)

def positions_plot(frames):
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

    x = remove_blank_spaces(x)
    x2 = remove_blank_spaces(x2)
    y = remove_blank_spaces(y)
    y2 = remove_blank_spaces(y2)

    x = moving_average(x, 15)
    y = moving_average(y, 15)
    x2 = moving_average(x2, 15)
    y2 = moving_average(y2, 15)

    ax.plot(x, label="manual")
    ax.plot(x2, label="automatic")
    ax.set_xlabel("Numer klatki")
    ax.set_ylabel("Pozycja x zawodnika")
    ax.legend()

    ax2.plot(y, label="manual")
    ax2.plot(y2, label="automatic")
    ax2.set_xlabel("Numer klatki")
    ax2.set_ylabel("Pozycja y zawodnika")
    # ax.set_xlim([200, 300])
    # ax.set_ylim([250, 400])
    ax2.legend()
    plt.tight_layout()
    plt.show()
    print(frames)

frames = pickler.unpickle_data("data/cache/baltyk_starogard_1.mp4_path_selector_middle.pik")
scatter_plot(frames, show_pitch=True)
positions_plot(frames)
