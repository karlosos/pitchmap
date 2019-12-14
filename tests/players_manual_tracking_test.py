import unittest
from manual_tracking import players


class PlayersTest(unittest.TestCase):
    def test_empty_list_after_create(self):
        pl = players.PlayersList(frames_length=10)
        self.assertEqual(len(pl.players), 10)
        self.assertEqual(pl.players[0], [])

    def test_adding_player(self):
        pl = players.PlayersList(frames_length=10)
        p = pl.add_player(player_id=1, position=(3, 2), color=1, frame_number=1)
        self.assertEqual(pl.players[1][0], p)

    def test_get_players_with_same_color(self):
        pl = players.PlayersList(frames_length=10)
        p1 = pl.add_player(player_id=1, position=(3, 2), color=1, frame_number=1)
        p2 = pl.add_player(player_id=2, position=(5, 1), color=1, frame_number=1)
        p3 = pl.add_player(player_id=3, position=(10, 12), color=2, frame_number=1)

        frame_number = 1
        color = 1
        players_with_same_color = pl.get_players_with_color(frame_number, color)
        self.assertEqual(len(players_with_same_color), 2)

        frame_number = 1
        color = 3
        players_with_same_color = pl.get_players_with_color(frame_number, color)
        self.assertEqual(len(players_with_same_color), 0)

    def test_get_nearest_player_with_same_color(self):
        pl = players.PlayersList(frames_length=10)
        p1 = pl.add_player(player_id=1, position=(3, 2), color=1, frame_number=1)
        p2 = pl.add_player(player_id=2, position=(6, 5), color=1, frame_number=1)
        p3 = pl.add_player(player_id=3, position=(10, 12), color=1, frame_number=1)
        searchable_players = [p1, p2, p3]
        position = (9, 10)
        nearest_player, distance = pl.get_nearest_player(searchable_players, position)
        self.assertEqual(nearest_player, p3)

    def test_default_player_id(self):
        pl = players.PlayersList(frames_length=10)
        p1 = pl.add_player(player_id=1, position=(3, 2), color=1, frame_number=1)
        p2 = pl.add_player(player_id=2, position=(6, 5), color=1, frame_number=1)

        position = (3, 3)
        default_id = pl.default_player_id(position=position, frame_number=2)
        self.assertEqual(default_id, p1.id)

        position = (120, 4)
        default_id = pl.default_player_id(position=position, frame_number=2)
        self.assertEqual(default_id, None)


if __name__ == '__main__':
    unittest.main()
