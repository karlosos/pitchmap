import numpy as np


def interpolate_transformation_matrices(n, m, H_n, H_m):
    """
    :param n: start frame index
    :param m: end frame index
    :param H_n: transformation matrix in n-th frame
    :param H_m: transformation matrix in m-th frame
    :return: array of transformation matrices for every frame between n and m
    """
    H = np.zeros([3, 3, int(m - n)])
    for i in range(3):
        for j in range(3):
            for k in range(int(m - n)):
                H[i, j, k] = H_n[i, j] + k * (H_m[i, j] - H_n[i, j]) / (m - n)
    return H
