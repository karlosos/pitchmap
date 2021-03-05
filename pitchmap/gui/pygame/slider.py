import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)


class Slider:
    def __init__(self, val, maxi, mini):
        self.val = val  # start value
        self.maxi = maxi  # maximum at slider position right
        self.mini = mini  # minimum at slider position left
        self.xpos = 57  # x-location on screen
        self.ypos = 457

        self.__height = 30
        self.__width = 600

        self.surf = pygame.surface.Surface((600, self.__height))
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction

        # Static graphics - slider background #
        self.surf.fill(WHITE)
        pygame.draw.rect(self.surf, BLACK, [0, 0, self.__width, self.__height], 3)
        white_width = self.__width-20
        white_height = self.__height-30

        pygame.draw.rect(self.surf, BLACK, [10, int((self.__height-white_height)/2), white_width, white_height], 0)

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, BLACK, (10, 10), 6, 0)
        pygame.draw.circle(self.button_surf, ORANGE, (10, 10), 4, 0)
        self.button_rect = self.button_surf.get_rect(center=(0, 0))

    def draw(self, win):
        """ Combination of static and dynamic graphics in a copy of
        the basic slide surface
        """
        # static
        surf = self.surf.copy()

        # dynamic
        pos = (10+int((self.val-self.mini)/(self.maxi-self.mini)*580), int(self.__height/2))
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position

        # screen
        win.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
        The dynamic part; reacts to movement of the slider button.
        """
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 580 * (self.maxi - self.mini) + self.mini
        if self.val < self.mini:
            self.val = self.mini
        if self.val > self.maxi:
            self.val = self.maxi

        print(f"slider position: {self.val}")

    def set_value(self, value):
        self.val = value
