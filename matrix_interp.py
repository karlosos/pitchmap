import numpy as np


def transformation_matrix_interpolation(n, m, H_n, H_m):
    H = np.zeros([3, 3, m-n])
    for i in range(3):
        for j in range(3):
            for k in range(m-n):
                H[i, j, k] = H_n[i, j] + k * (H_m[i, j]-H_n[i, j])/(m-n)
    return H


if __name__ == '__main__':
    n = 2
    m = 5
    H_n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    H_m = np.array([[5, 5, 5], [5, 5, 6], [7, 8, 9]])
    H = transformation_matrix_interpolation(n, m, H_n, H_m)
    for k in range(m-n):
        print(H[:, :, k])
