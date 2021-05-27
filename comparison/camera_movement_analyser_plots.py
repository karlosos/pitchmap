"""
This script is made for printing camera movement models from previously gathered data and gather frames from input video.
"""

from pitchmap.cache_loader import pickler
from pitchmap.frame.loader import FrameLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def plot(x_cum_sum):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_ylabel("wychylenie kamery \n względem początku")
    ax.set_xlabel("numer klatki")
    ax.plot(x_cum_sum)
    ax.set_xticks(np.arange(0, len(x_cum_sum), 50))
    plt.tight_layout()
    plt.savefig(f"./data/experiments/movement_analysis/camera_movement.pdf")
    plt.show()


def plot_characteristic_frames(camera_angles):
    min_peaks, _ = find_peaks(-camera_angles, width=3)
    max_peaks, _ = find_peaks(camera_angles, width=3)
    characteristic_frames = np.concatenate(([0], min_peaks, max_peaks, [len(camera_angles) - 1]))
    characteristic_frames = np.sort(characteristic_frames)
    characteristic_frames = np.array([0, 162, 245, 268, 460])
    print(characteristic_frames)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_ylabel("wychylenie kamery \n względem początku")
    ax.set_xlabel("numer klatki")
    ax.plot(x_cum_sum)
    ax.scatter(characteristic_frames, x_cum_sum[characteristic_frames], s=50, color='black')
    for idx, frame in enumerate(characteristic_frames):
        ax.annotate(idx+1, (frame, x_cum_sum[frame] - 20))
    ax.set_xticks(np.arange(0, len(x_cum_sum), 50))
    ax.set_ylim(-210, 10)
    plt.tight_layout()
    plt.savefig(f"./data/experiments/movement_analysis/camera_movement_peaks.pdf")
    plt.show()


if __name__ == '__main__':
    input_video = "Baltyk_Koszalin_05_06.mp4"
    fl = FrameLoader(file_name=input_video)
    x_cum_sum = pickler.unpickle_data(file_name=f"./data/cache/{input_video}_CameraMovementAnalyser.pik")

    # Plot clean chart
    # plot(x_cum_sum)

    # Save frames
    # frame_indexes = np.arange(0, len(x_cum_sum), 50)
    # frames = fl.get_frames(frame_indexes)
    # for idx, frame in enumerate(frames):
    #     plt.imshow(frame[..., ::-1])
    #     plt.tight_layout()
    #     plt.axis('off')
    #     plt.imsave(f"./data/experiments/movement_analysis/frame_{frame_indexes[idx]}.png", frame[..., ::-1])
    #     plt.show()

    plot_characteristic_frames(x_cum_sum)
