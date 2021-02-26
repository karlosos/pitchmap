import numpy as np


INIT_HOMO_MAPPER = {
    0: [3, 3],
    1: [3, 66],
    2: [51, 65],
    3: [3, 117],
    4: [17, 117],
    5: [3, 203],
    6: [17, 203],
    7: [3, 255],
    8: [51, 254],
    9: [3, 317],
    10: [160, 3],
    11: [160, 160],
    12: [160, 317],
    13: [317, 3],
    14: [317, 66],
    15: [270, 66],
    16: [317, 118],
    17: [304, 118],
    18: [317, 203],
    19: [304, 203],
    20: [317, 255],
    21: [271, 255],
    22: [317, 317],
    23: [51, 128],
    24: [51, 193],
    25: [161, 118],
    26: [161, 203],
    27: [270, 128],
    28: [269, 192],
}

MY_HOMO_MAPPER = {
    0: (37, 34),
    1: (37, 114),
    2: (115, 114),
    3: (37, 167),
    4: (64, 167),
    5: (37, 252),
    6: (64, 252),
    7: (37, 308),
    8: (115, 308),
    9: (37, 385),
    10: (300, 34),
    11: (300, 210),
    12: (300, 385),
    13: (560, 34),
    14: (560, 114),
    15: (484, 114),
    16: (560, 166),
    17: (535, 166),
    18: (560, 252),
    19: (535, 253),
    20: (560, 306),
    21: (482, 306),
    22: (561, 385),
    23: (115, 178),
    24: (115, 245),
    25: (300, 164),
    26: (300, 255),
    27: (485, 178),
    28: (485, 245)
}


def _get_keypoints_from_mask(mask, treshold=0.9):
    """From a list of mask, compute the mapping of each keypoints to their location

    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        keypoints: Dict, mapping each keypoint id to its location
    Raises:

    """
    keypoints = {}
    indexes = np.argwhere(mask[:, :, :-1] > treshold)
    for indx in indexes:
        id_kp = indx[2]
        if id_kp in keypoints.keys():
            keypoints[id_kp][0].append(indx[0])
            keypoints[id_kp][1].append(indx[1])
        else:
            keypoints[id_kp] = [[indx[0]], [indx[1]]]

    for id_kp in keypoints.keys():
        mean_x = np.mean(np.array(keypoints[id_kp][0]))
        mean_y = np.mean(np.array(keypoints[id_kp][1]))
        keypoints[id_kp] = [mean_y, mean_x]
    return keypoints


def collinear(p0, p1, p2, epsilon=0.001):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < epsilon


def _points_from_mask(mask, treshold=0.9):
    """From a list of mask, compute src and dst points from the image and the 2D view of the image

    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        src_pts, dst_pts: Location of src and dst related points
    Raises:

    """
    list_ids = []
    src_pts, dst_pts = [], []
    available_keypoints = _get_keypoints_from_mask(mask, treshold)
    for id_kp, v in available_keypoints.items():
        src_pts.append(v)
        dst_pts.append(MY_HOMO_MAPPER[id_kp])
        list_ids.append(id_kp)
    src, dst = np.array(src_pts), np.array(dst_pts)

    ### Final test : return nothing if 3 points are colinear and the src has just 4 points
    test_colinear = False
    if len(src) == 4:
        if collinear(dst_pts[0], dst_pts[1], dst_pts[2]) or collinear(dst_pts[0], dst_pts[1], dst_pts[3]) or collinear(
                dst_pts[1], dst_pts[2], dst_pts[3]):
            test_colinear = True
    src = np.array([]) if test_colinear else src
    dst = np.array([]) if test_colinear else dst

    return src, dst