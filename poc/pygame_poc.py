import pygame, math, sys
pygame.init()

X = 900  # screen width
Y = 600  # screen height

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

flow = False  # controls type of color flow


class Gradient():
    def __init__(self, palette, maximum):
        self.COLORS = palette
        self.N = len(self.COLORS)
        self.SECTION = maximum // (self.N - 1)

    def gradient(self, x):
        """
        Returns a smooth color profile with only a single input value.
        The color scheme is determinated by the list 'self.COLORS'
        """
        i = x // self.SECTION
        fraction = (x % self.SECTION) / self.SECTION
        c1 = self.COLORS[i % self.N]
        c2 = self.COLORS[(i+1) % self.N]
        col = [0, 0, 0]
        for k in range(3):
            col[k] = (c2[k] - c1[k]) * fraction + c1[k]
        return col


def wave(num):
    """
    The basic calculating and drawing function.
    The internal function is 'cosine' >> (x, y) values.

    The function uses slider values to variate the output.
    Slider values are defined by <slider name>.val
    """
    for x in range(0, X+10, int(jmp.val)):

        # Calculations #
        ang_1 = (x + num) * math.pi * freq.val / 180
        ang_2 = ang_1 - phase.val
        cos_1 = math.cos(ang_1)
        cos_2 = math.cos(ang_2)

        y_1 = int(cos_1 * size.val) + 250
        y_2 = int(cos_2 * size.val) + 250

        radius_1 = int(pen.val + math.sin(ang_1 + focus.val) * pen.val / 2)
        radius_2 = int(pen.val + math.sin(ang_2 + focus.val) * pen.val / 2)

        # Drawing #
        if radius_1 > radius_2:  # draw the smaller circle before the larger one
            pygame.draw.circle(screen, xcolor(int(x + X//2) + num * flow), (x, y_2), radius_2, 0)
            pygame.draw.circle(screen, xcolor(x + num * flow), (x, y_1), radius_1, 0)
        else:
            pygame.draw.circle(screen, xcolor(x + num * flow), (x, y_1), radius_1, 0)
            pygame.draw.circle(screen, xcolor(int(x + X//2) + num * flow), (x, y_2), radius_2, 0)


class Slider():
    def __init__(self, name, val, maxi, mini, pos):
        self.val = val  # start value
        self.maxi = maxi  # maximum at slider position right
        self.mini = mini  # minimum at slider position left
        self.xpos = pos  # x-location on screen
        self.ypos = 550
        self.surf = pygame.surface.Surface((100, 50))
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction

        self.txt_surf = font.render(name, 1, BLACK)
        self.txt_rect = self.txt_surf.get_rect(center=(50, 15))

        # Static graphics - slider background #
        self.surf.fill((100, 100, 100))
        pygame.draw.rect(self.surf, GREY, [0, 0, 100, 50], 3)
        pygame.draw.rect(self.surf, ORANGE, [10, 10, 80, 10], 0)
        pygame.draw.rect(self.surf, WHITE, [10, 30, 80, 5], 0)

        self.surf.blit(self.txt_surf, self.txt_rect)  # this surface never changes

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, BLACK, (10, 10), 6, 0)
        pygame.draw.circle(self.button_surf, ORANGE, (10, 10), 4, 0)

    def draw(self):
        """ Combination of static and dynamic graphics in a copy of
    the basic slide surface
    """
        # static
        surf = self.surf.copy()

        # dynamic
        pos = (10+int((self.val-self.mini)/(self.maxi-self.mini)*80), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position

        # screen
        screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
    The dynamic part; reacts to movement of the slider button.
    """
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 80 * (self.maxi - self.mini) + self.mini
        if self.val < self.mini:
            self.val = self.mini
        if self.val > self.maxi:
            self.val = self.maxi


font = pygame.font.SysFont("Verdana", 12)
screen = pygame.display.set_mode((X, Y))
clock = pygame.time.Clock()

COLORS = [MAGENTA, RED, YELLOW, GREEN, CYAN, BLUE]
xcolor = Gradient(COLORS, X).gradient

pen = Slider("Pen", 10, 15, 1, 25)
freq = Slider("Freq", 1, 3, 0.2, 150)
jmp = Slider("Jump", 10, 20, 1, 275)
size = Slider("Size", 200, 200, 20, 400)
focus = Slider("Focus", 0, 6, 0, 525)
phase = Slider("Phase", 3.14, 6, 0.3, 650)
speed = Slider("Speed", 50, 150, 10, 775)
slides = [pen, freq, jmp, size, focus, phase, speed]

num = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for s in slides:
                if s.button_rect.collidepoint(pos):
                    s.hit = True
        elif event.type == pygame.MOUSEBUTTONUP:
            for s in slides:
                s.hit = False

    # Move slides
    for s in slides:
        if s.hit:
            s.move()

    # Update screen
    screen.fill(BLACK)
    num += 2
    wave(num)

    for s in slides:
        s.draw()

    pygame.display.flip()
    clock.tick(speed.val)