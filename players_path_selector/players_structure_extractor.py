def get_players_positions(players, frame_number):
    """
    Extract players positions from players structure (dict of frames)
    :param players: dict where key is frame and value is list of players
    :param frame_number: from which frame get players positions
    :return: list of players positions for given frame
    """
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


def get_players_ids(players, frame_number):
    """
    Extract players ids from players structure (dict of frames)
    :param players: dict where key is frame and value is list of players
    :param frame_number: from which frame get players ids
    :return: list of players ids for given frame
    """
    try:
        players = players[frame_number]
    except IndexError:
        players = []

    if players:
        if type(players) is dict:
            players = list(players.values())
        ids = list(map(lambda player: player.id, players))
    else:
        ids = []
    return ids


def get_players_team_ids(players, frame_number):
    """
    Extract players team ids from players structure (dict of frames)
    :param players: dict where key is frame and value is list of players
    :param frame_number: from which frame get team ids
    :return: list of team ids for given frame
    """
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
    return colors
