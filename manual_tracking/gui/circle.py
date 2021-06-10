import pygame


class Circle:
    def __init__(self, x, y, radius, color, start_x=740, start_y=210):
        self.x = x
        self.y = y
        self.radius = radius
        self.thickness = self.radius
        self.color = color
        self.start_x = start_x
        self.start_y = start_y

    def draw(self, win):
        pygame.draw.circle(win, self.color, [self.start_x + self.x, self.start_y + self.y], self.radius, self.thickness)

    def is_over(self, pos):
        size = self.radius
        real_x = self.x + self.start_x
        real_y = self.y + self.start_y
        if real_x - size < pos[0] < real_x + size:
            if real_y - size < pos[1] < real_y + size:
                return True

        return False

    def get_relative_pos(self, pos):
        x = pos[0] - self.x
        y = pos[1] - self.y
        return x, y


class PlayerCircle(Circle):
    def __init__(self, player, radius, start_x=740, start_y=210):
        team_color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        x = player.position[0]
        y = player.position[1]
        color = team_color[player.color]
        super().__init__(x, y, radius, color, start_x, start_y)
        self.player = player
        self.font = pygame.font.SysFont("Times New Roman", 18)

    def reset_highlight(self):
        self.thickness = 2

    def highlight(self):
        self.thickness = self.radius

    def draw(self, win):
        pygame.draw.circle(win, self.color, [self.start_x + self.x, self.start_y + self.y], self.radius, self.thickness)
        txt_surface = self.font.render(str(self.player.id), True, self.color)
        win.blit(txt_surface, (self.start_x + self.x + 5, self.start_y + self.y + 5))


class LastPlayerCircle(Circle):
    def __init__(self, player, radius, start_x=740, start_y=210):
        team_color = [(0, 120, 0), (120, 0, 0), (0, 0, 120)]
        x = player.position[0]
        y = player.position[1]
        color = team_color[player.color]
        super().__init__(x, y, radius, color, start_x, start_y)
        self.player = player
        self.font = pygame.font.SysFont("Times New Roman", 18)

    def reset_highlight(self):
        self.thickness = 2

    def highlight(self):
        self.thickness = self.radius

    def draw(self, win):
        pygame.draw.circle(win, self.color, [self.start_x + self.x, self.start_y + self.y], self.radius, self.thickness)
        txt_surface = self.font.render(str(self.player.id), True, self.color)
        win.blit(txt_surface, (self.start_x + self.x + 5, self.start_y + self.y + 5))


class CalibrationCircle(Circle):
    def __init__(self, index, pos, point, point_index, radius=5):
        """
        :param index: index of characteristic point
        :param pos: position of characterstic point, absolute position in windows surface
        :param point: reference to point object stored in calibration
        :param point_index: index of which point it is. 0 is point on pitch, 1 is point on model
        :param radius: raius of circle
        """
        color = (183, 31, 54)
        x = pos[0]
        y = pos[1]
        super().__init__(x, y, radius, color, 0, 0)
        self.id = index
        self.font = pygame.font.SysFont("Times New Roman", 18)
        self.point = point
        self.point_index = point_index
        self.hit = False

    def reset_highlight(self):
        self.thickness = 2

    def highlight(self):
        self.thickness = self.radius

    def draw(self, win):
        pygame.draw.circle(win, self.color, [self.start_x + self.x, self.start_y + self.y], self.radius, self.thickness)
        txt_surface = self.font.render(str(self.id), True, self.color)
        win.blit(txt_surface, (self.start_x + self.x + 5, self.start_y + self.y + 5))

    def move(self):
        x_diff = self.x - pygame.mouse.get_pos()[0]
        y_diff = self.y - pygame.mouse.get_pos()[1]
        print(self.point)
        self.point[self.point_index] = (self.point[self.point_index][0] - x_diff,
                                        self.point[self.point_index][1] - y_diff)
        print(self.point)
        self.x = pygame.mouse.get_pos()[0]
        self.y = pygame.mouse.get_pos()[1]

        print(f"position: {self.x} {self.y}")
