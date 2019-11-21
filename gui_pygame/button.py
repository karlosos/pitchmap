import pygame

class Button:
    COLOR_DISABLED = (140, 16, 0)
    COLOR_ENABLED = (0, 140, 33)
    COLOR_STANDARD = (27, 27, 27)
    COLOR_HOVER = (57, 57, 57)

    def __init__(self, x, y, width, height, text=''):
        self.color = self.COLOR_STANDARD
        self.state_color = self.COLOR_STANDARD
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.is_hover = False

    def draw(self, win, outline=None):
        if outline:
            pygame.draw.rect(win, outline, (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)

        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height), 0)

        if self.text != '':
            font = pygame.font.SysFont("Verdana", 12)
            text = font.render(self.text, 1, (245, 245, 245))
            win.blit(text, (self.x + (self.width / 2 - text.get_width() / 2),
                            self.y + (self.height / 2 - text.get_height() / 2)))

    def is_over(self, pos):
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                return True

        return False
