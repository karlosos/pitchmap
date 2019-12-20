from .button import Button

import pygame

TEAM_COLOR = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

class ColorSelector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 104
        self.height = 36
        self.surf = pygame.surface.Surface((self.width, self.height))
        self._button_0 = Button(x=2, y=2, width=32, height=32, text="0")
        self._button_1 = Button(x=36, y=2, width=32, height=32, text="1")
        self._button_2 = Button(x=70, y=2, width=32, height=32, text="2")

        self._button_0.color = TEAM_COLOR[0]
        self._button_1.color = TEAM_COLOR[1]
        self._button_2.color = TEAM_COLOR[2]

    def draw(self, win, team_id=None):
        self.surf.fill((255, 255, 255))
        button_0_outline = None
        button_1_outline = None
        button_2_outline = None

        if team_id is not None:
            if team_id == 0:
                button_0_outline = (0, 0, 0)
            elif team_id == 1:
                button_1_outline = (0, 0, 0)
            elif team_id == 2:
                button_2_outline = (0, 0, 0 )

        self._button_0.draw(self.surf, outline=button_0_outline)
        self._button_1.draw(self.surf, outline=button_1_outline)
        self._button_2.draw(self.surf, outline=button_2_outline)

        win.blit(self.surf, (self.x, self.y))

    def is_over(self, pos):
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                pos = self.get_relative_pos(pos)
                if self._button_0.is_over(pos):
                    return 0
                elif self._button_1.is_over(pos):
                    return 1
                elif self._button_2.is_over(pos):
                    return 2

        return None

    def get_relative_pos(self, pos):
        x = pos[0] - self.x
        y = pos[1] - self.y
        return x, y
