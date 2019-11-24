def input_point(key):
    if key == 115:  # s key
        return True
    else:
        return False


def input_exit(key):
    if key == 27:
        return True
    else:
        return False


def input_transform(key):
    if key == 116:  # t key
        return True
    else:
        return False


def input_test(key):
    if key == 117:  # u key
        return True
    else:
        return False


def key_pressed(key, pitch_map):
    if input_exit(key):
        return False
    elif input_point(key):
        pitch_map.start_calibration()
    elif input_transform(key):
        pitch_map.perform_transform()
    elif input_test(key):
        pitch_map.input_test()
    return True
