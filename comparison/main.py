import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

from pitchmap.cache_loader import pickler
from pitchmap.homography import calibrator
from pitchmap.players import structure
from comparison import heatmap
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse

def get_players_positions_from_frame(players, frame_number):
    try:
        players = players[frame_number]
    except IndexError:
        players = []

    if players:
        if type(players) is dict:
            players = list(players.values())
        positions = [player.position for player in players]
    else:
        positions = []

    return positions


def get_players_team_ids_from_frame(players, frame_number):
    # print(f"get colors from frame {frame_number}")
    try:
        players = players[frame_number]
    except IndexError:
        players = []

    if players:
        if type(players) is dict:
            players = list(players.values())
        colors = list(map(lambda player: player.calculate_real_color(), players))
    else:
        colors = []
    # print(f"colors: {colors}")
    return colors


class Camparator:
    def __init__(self):
        #self.file_detected_data = "data/cache/baltyk_kotwica_1.mp4_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"
        self.file_detected_data = "data/cache/baltyk_starogard_1.mp4_PlayersListComplex_CalibrationInteractorMiddlePoint.pik"
        #self.file_manual_data = "data/cache/baltyk_kotwica_1.mp4_manual_tracking.pik"
        self.file_manual_data = "data/cache/baltyk_starogard_1.mp4_manual_tracking.pik"

        self.pitch_model = cv2.imread('data/pitch_model.jpg')
        self.pitch_model = imutils.resize(self.pitch_model, width=600)

        players_detected, _, homographies_detected = pickler.unpickle_data(self.file_detected_data)
        players_list_manual, _, _ = pickler.unpickle_data(self.file_manual_data)

        players_detected_transformed = []
        length_homographies_detected = len(homographies_detected)
        for i, players in enumerate(players_detected):
            players_positions = get_players_positions_from_frame(players_detected, i)
            homography = homographies_detected[i] if i < length_homographies_detected else homographies_detected[-1]
            players_2d_positions = calibrator.Calibrator.transform_to_2d(players_positions, homography)
            players_colors = get_players_team_ids_from_frame(players_detected, i)

            players_in_frame = []
            for i, position in enumerate(players_2d_positions):
                player = structure.PlayerSimple((position[0], position[1]), players_colors[i])
                players_in_frame.append(player)
            players_detected_transformed.append(players_in_frame)

        self.players_detected = players_detected_transformed
        self.players_manual = players_list_manual.players

    def generate_heat_map(self, players):
        pitch_model_size = (421, 600)
        pitch_heat_map = np.zeros(pitch_model_size)
        for frame in players:
            for player in frame:
                if player.id == 21:
                    print("here we go")
                self.add_to_heat_map(pitch_heat_map, player.position, size=5, value=10)
                self.add_to_heat_map(pitch_heat_map, player.position, size=8, value=5)
                self.add_to_heat_map(pitch_heat_map, player.position, size=15, value=2)
                self.add_to_heat_map(pitch_heat_map, player.position, size=20, value=1)

        heatmap.add(self.pitch_model, pitch_heat_map, alpha=0.8, display=False, cmap='jet')
        return pitch_heat_map

    @staticmethod
    def add_to_heat_map(heat_map, position, size, value):
        x = int(position[0])
        y = int(position[1])
        size_heat_map = heat_map.shape
        if x >= size_heat_map[1]:
            x = size_heat_map[1]-1
        if y >= size_heat_map[0]:
            y = size_heat_map[0]-1

        yy = np.arange(0, heat_map.shape[0])
        xx = np.arange(0, heat_map.shape[1])
        mask = (xx[np.newaxis, :] - x) ** 2 + (yy[:, np.newaxis] - y) ** 2 < size ** 2
        heat_map[mask] += value


if __name__ == '__main__':
    c = Camparator()
    GENERATING = False

    if GENERATING:
        heat_maps = []
        manual_heatmap = c.generate_heat_map(c.players_manual)
        heat_maps.append(manual_heatmap)

        for idx, x in enumerate([700, 400, 200, 100, 80, 30, 10]):
            detected_heatmap = c.generate_heat_map(c.players_detected[:-x])
            heat_maps.append(detected_heatmap)

        pickler.pickle_data(heat_maps, f"data/cache/heatmaps_1.pik")
    else:
        heat_maps = pickler.unpickle_data(f"data/cache/heatmaps_1.pik")

    manual_heat_map = heat_maps[0]
    heat_map_min = np.min(manual_heat_map)
    heat_map_max = np.max(manual_heat_map)
    manual_heat_map = (manual_heat_map-heat_map_min)/(heat_map_max-heat_map_min)
    cv2.imwrite(f'99.png', manual_heat_map * 255)

    psnr = []
    mse = []
    ssim = []

    for idx, heat_map in enumerate(heat_maps):
        heat_map_min = np.min(heat_map)
        heat_map_max = np.max(heat_map)
        heat_map = (heat_map - heat_map_min) / (heat_map_max - heat_map_min)
        cv2.imwrite(f'{idx}.png', heat_map * 255)
        print(f"mse: {compare_mse(manual_heat_map, heat_map)}")
        mse.append(compare_mse(manual_heat_map, heat_map))
        print(f"psnr: {compare_psnr(manual_heat_map, heat_map)}")
        psnr.append(compare_psnr(manual_heat_map, heat_map))
        (score, diff) = compare_ssim(manual_heat_map, heat_map, full=True)
        diff = (diff * 255).astype("uint8")
        # 6. You can print only the score if you want
        print("SSIM: {}".format(score))
        ssim.append(score)

    plt.figure()
    #plt.plot(psnr)
    #plt.plot(mse)
    plt.plot(ssim)
    plt.show()