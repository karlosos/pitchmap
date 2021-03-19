from pitchmap.cache_loader.pickler import unpickle_data
import numpy as np


def main():
    (characteristic_homographies, guessed_homographies, flags, false_indices) = unpickle_data("./data/cache/characteristic_frames.pickle")
    print(flags)
    good_distances = []
    bad_distances = []
    positions = [[200, 200, 1], [10, 10, 1], [10, 200, 1], [200, 10, 1], [600, 10, 1], [10, 300, 1]]
    for i in range(len(flags)):
        homo_pred = characteristic_homographies[i]
        homo_guessed = guessed_homographies[i]
        if homo_pred is not None:
            distances = []
            for pos in positions:
                pos_pred = np.dot(pos, homo_pred)
                pos_guessed = np.dot(pos, homo_guessed)
                dist = np.linalg.norm(pos_pred - pos_guessed)
                distances.append(dist)

            print(f"{np.mean(distances)/1000} {flags[i]}")

            if flags[i]:
                good_distances.append(np.mean(distances))
            else:
                bad_distances.append(np.mean(distances))

    print("Good distances:")
    print("Minimum:", np.min(good_distances))
    print("Maximum:", np.max(good_distances))
    print("Mean:", np.mean(good_distances))
    print()
    print("Bad distances:")
    print("Minimum:", np.min(bad_distances))
    print("Maximum:", np.max(bad_distances))
    print("Mean:", np.mean(bad_distances))




if __name__ == '__main__':
    main()