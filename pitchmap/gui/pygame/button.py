import pygame

class Button:
    COLOR_DISABLED = (140, 16, 0)
    COLOR_ENABLED = (40, 167, 69)
    COLOR_STANDARD = (255, 255, 255)
    COLOR_HOVER = (220, 220, 220)

    def __init__(self, x, y, width, height, text=''):
        self.color = self.COLOR_STANDARD
        self.state_color = self.COLOR_STANDARD
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.is_hover = False

    def draw(self, win, outline=True):
        if outline:
            pygame.draw.rect(win, (18, 18, 18), (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)

        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height), 0)

        if self.text != '':
            #font = pygame.font.SysFont("Verdana", 12)
            font = pygame.font.Font("data/fonts/weblysleekuisb.ttf", 15)
            text = font.render(self.text, 1, (18, 18, 18))
            win.blit(text, (self.x + (self.width / 2 - text.get_width() / 2),
                            self.y + (self.height / 2 - text.get_height() / 2)))

    def is_over(self, pos):
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                return True

        return False

    def update(self, enabled=None):
        if enabled is not None:
            if enabled:
                self.state_color = self.COLOR_ENABLED
            else:
                self.state_color = self.get_hover_color()
        else:
            self.state_color = self.get_hover_color()
        self.color = self.state_color

    def get_hover_color(self):
        if self.is_hover:
            return self.COLOR_HOVER
        else:
            return self.COLOR_STANDARD

    def update_hover(self, pos):
        if self.is_over(pos):
            self.is_hover = True
        else:
            self.is_hover = False