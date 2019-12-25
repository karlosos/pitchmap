from pitchmap.cache_loader import pickler
from players_path_selector import players_structure_extractor as pse
from pitchmap.players import structure
from players_path_selector import transformation


class DataLoader:
    def __init__(self):
        self.players_detected = []
        self.players_manual = []
        self.homographies_detected = []
        self.homographies_manual = []

    def load_data(self, file_detected_data, file_manual_data):
        players_detected, _, homographies_detected = pickler.unpickle_data(file_detected_data)
        players_list_manual, homographies_manual, _ = pickler.unpickle_data(file_manual_data)

        self.homographies_detected = {}
        players_detected_transformed = []
        length_homographies_detected = len(homographies_detected)
        for i, players in enumerate(players_detected):
            players_positions = pse.get_players_positions(players_detected, i)
            homography = homographies_detected[i] if i < length_homographies_detected else homographies_detected[-1]
            self.homographies_detected[i] = homography

            players_2d_positions = transformation.bulk_transform_to_2d(players_positions, homography)
            players_colors = pse.get_players_team_ids(players_detected, i)
            players_ids = pse.get_players_ids(players_detected, i)

            players_in_frame = []
            for i, position in enumerate(players_2d_positions):
                player = structure.PlayerSimple((position[0], position[1]), players_colors[i])
                player.id = players_ids[i]
                players_in_frame.append(player)
            players_detected_transformed.append(players_in_frame)

        self.players_detected = players_detected_transformed
        self.players_manual = players_list_manual.players

        self.homographies_manual = homographies_manual
