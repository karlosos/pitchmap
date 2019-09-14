from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_colors(colors, hsv, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.asarray(colors)
    ax.scatter(hsv[:, 0], hsv[:, 1], hsv[:, 2], c=labels, depthshade=False)
    plt.show()
    print(colors)
