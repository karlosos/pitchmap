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
