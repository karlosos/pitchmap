class Calibrator:
    def __init__(self):
        self.enabled = False
        self.points = {}

        self.current_point = None

    def toggle_enabled(self):
        if not self.enabled:
            self.current_point = None

        self.enabled = not self.enabled

    def add_point_main_window(self, pos):
        if self.current_point is None:
            self.current_point = pos
            index = len(self.points) + 1
            return index
        else:
            return False

    def add_point_model_window(self, pos):
        if self.current_point is not None:
            index = len(self.points) + 1
            self.points[index] = (self.current_point, pos)
            print(self.points)
            self.current_point = None
            return index
        else:
            return False

    def get_points_count(self):
        return len(self.points)
