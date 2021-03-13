"""
Find characteristic frames based on optical flow analysis
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks


def find_characteristic_frames(x):
    min_peaks, _ = find_peaks(-x, width=3)
    max_peaks, _ = find_peaks(x, width=3)
    return np.concatenate((min_peaks, max_peaks))


def main():
    x = np.load("./data/experiments/camera_movement.npy")
    characteristic_frames = find_characteristic_frames(x)
    plt.plot(characteristic_frames, x[characteristic_frames], 'x')
    plt.plot(x)
    plt.show()


if __name__ == '__main__':
    main()
