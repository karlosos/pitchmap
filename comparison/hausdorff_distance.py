import numpy as np

def modified_hausdorff_distance(A, B):
    """
    Calculate modified hausdorff distance for 2D points sets

    A modified Hausdorff distance for object matching
    doi: 10.1109/ICPR.1994.576361

    :param A: 2D points set
    :param B: 2D points set
    :return: MDH - undirected distance
    """
    return np.max([directed_distance(A, B), directed_distance(B, A)])

def directed_distance(A, B):
    """
    Calculate directed distance between two sets of 2D points
    :param A: 2D points set
    :param B: 2D points set
    :return: directed sitance
    """
    Na = len(A)
    sum = 0
    for a in A:
        sum += distance_between_point_and_set(a, B)
    return sum/Na

def distance_between_point_and_set(a, B):
    """
    Calculate distance between point and a set

    :param a: is a 2D point
    :param B: is set of 2D points
    :return: distance between point and set
    """
    distances = [euclid_distance(a, b) for b in B]
    return np.min(distances)


def euclid_distance(a, b):
    """
    Calculate euclid distance between two 2D points

    :param a: 2D point
    :param b: 2D point
    :return: euclid distance
    """
    dist = np.linalg.norm(a - b)
    return dist
