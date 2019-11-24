import unittest

from pitchmap.players.structure import PlayersListComplex


class PlayersListTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = PlayersListComplex(100)

    def test_findPlayerId_onePlayerInListNear_returnNearestPlayer(self):
        p = self.p
        p.add_player(player_id=None, position=(90, 140), color=1, frame_number=0)
        added_player = p.add_player(player_id=None, position=(200, 200), color=1, frame_number=0)

        found_player_id = p.find_player_id(position=(199, 199), color=1, frame_number=1)

        self.assertEqual(added_player.id, found_player_id)

    def test_findPlayerId_newPlayerNotExistButNearAnother_returnNone(self):
        p = self.p
        added_player = p.add_player(player_id=None, position=(200, 200), color=1, frame_number=0)

        p.add_player(player_id=added_player.id, position=(201, 200), color=1, frame_number=1)
        found_player_id = p.find_player_id(position=(199, 199), color=1, frame_number=1)

        self.assertEqual(None, found_player_id)

    def test_findPlayerId_twoPlayersCloseSameColor_returnCloser(self):
        p = self.p
        further_player = p.add_player(player_id=None, position=(205, 205), color=1, frame_number=0)
        closer_player = p.add_player(player_id=None, position=(200, 200), color=1, frame_number=0)

        found_player_id = p.find_player_id(position=(199, 199), color=1, frame_number=1)

        self.assertEqual(closer_player.id, found_player_id)

    # def test_findPlayerId_twoPlayersCloseSameColorSecond_returnCloser(self):
    #     p = self.p
    #     closer_player = p.add_player(id=None, position=(205, 205), color=1, frame_number=0)
    #     further_player = p.add_player(id=None, position=(200, 200), color=1, frame_number=0)
    #
    #     found_player_id = p.find_player_id(position=(199, 199), color=1, frame_number=1)
    #
    #     self.assertEqual(closer_player.id, found_player_id)


class PlayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pl = PlayersListComplex(100)

    def test_calculateRealColor_colorInGivenFrame_returnGivenFrame(self):
        p = self.pl.add_player(player_id=None, position=(0, 0), color=0, frame_number=0)
        self.assertEqual(p.calculate_real_color(), 0)

    def test_calculateRealColor_colorInPreviousFrames_returnGivenFrame(self):
        p = self.pl.add_player(player_id=None, position=(0, 0), color=0, frame_number=0)
        self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=1)
        self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=2)
        p = self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=3)
        self.assertEqual(p.calculate_real_color(), 0)

    def test_calculateRealColor_multipleColorsInPreviousFrames_returnGivenFrame(self):
        p = self.pl.add_player(player_id=None, position=(0, 0), color=0, frame_number=0)
        self.pl.clone_player(player=p, position=(0, 0), color=1, frame_number=1)
        self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=2)
        p = self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=3)
        self.assertEqual(p.calculate_real_color(), 0)

    def test_calculateRealColor_moreMultipleColorsInPreviousFrames_returnGivenFrame(self):
        p = self.pl.add_player(player_id=None, position=(0, 0), color=0, frame_number=0)
        self.pl.clone_player(player=p, position=(0, 0), color=1, frame_number=1)
        self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=2)
        self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=3)
        p = self.pl.clone_player(player=p, position=(0, 0), color=1, frame_number=4)
        self.assertEqual(p.calculate_real_color(), 0)

    def test_calculateRealColor_differentValueMoreMultipleColorsInPreviousFrames_returnGivenFrame(self):
        p = self.pl.add_player(player_id=None, position=(0, 0), color=1, frame_number=0)
        self.pl.clone_player(player=p, position=(0, 0), color=2, frame_number=1)
        self.pl.clone_player(player=p, position=(0, 0), color=1, frame_number=2)
        self.pl.clone_player(player=p, position=(0, 0), color=0, frame_number=3)
        p = self.pl.clone_player(player=p, position=(0, 0), color=1, frame_number=4)
        self.assertEqual(p.calculate_real_color(), 1)


if __name__ == '__main__':
    unittest.main()
