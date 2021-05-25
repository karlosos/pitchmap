import numpy as np
import cv2
from matplotlib import pyplot as plt

import imutils
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


def pitch_plot(positions, show_pitch=False, smooth=False):
    names = ["Wzorzec", "Automatyczne", "2 klatki", "3 klatki"]

    fig, ax = plt.subplots()
    pitch_model = cv2.imread('data/pitch_model.jpg')
    pitch_model = imutils.resize(pitch_model, width=600)

    if show_pitch:
        ax.imshow(pitch_model)

    processed_positions = []
    for idx in range(len(positions)):
        x = positions[idx][:, 0]
        y = positions[idx][:, 1]

        if smooth:
            x = moving_average(x, 10)
            y = moving_average(y, 10)

        processed_positions.append([x, y])

    for idx in range(len(processed_positions)):
        name = names[idx]
        x = processed_positions[idx][0]
        y = processed_positions[idx][1]
        if idx == 0:
            ax.scatter(x, y, label=name, s=3.5, color='black')
        else:
            ax.scatter(x, y, label=name, s=3.5)

    plt.legend()
