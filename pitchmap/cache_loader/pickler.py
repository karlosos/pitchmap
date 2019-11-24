import pickle
import os


def unpickle_data(file_name):
    f = open(file_name, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def pickle_data(data, file_name):
    print(file_name)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()
