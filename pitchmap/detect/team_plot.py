from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot(colors, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.asarray(colors)
    ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=labels, depthshade=False)

    plt.show()
    print(colors)
