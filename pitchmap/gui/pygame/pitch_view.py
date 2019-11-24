import pygame
import cv2


class PitchView:
    def __init__(self, x=57, y=57, width=600, height=600):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.frame_x = x
        self.frame_y = y
        self.frame_width = width
        self.frame_height = height

    def draw(self, win, frame):
        pygame.draw.rect(win, (64, 64, 64), (self.x, self.y, self.width, self.height), 0)

        frame_size = frame.shape[1::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.image.frombuffer(rgb_frame, frame_size, 'RGB')
        self.frame_y = int(self.y + (self.height - frame_size[1])/2)
        self.frame_x = self.x
        self.frame_height = frame_size[1]
        self.frame_width = frame_size[0]
        win.blit(pygame_frame, (self.x, self.frame_y))

    def is_over(self, pos):
        if self.frame_x < pos[0] < self.frame_x + self.frame_width:
            if self.frame_y < pos[1] < self.frame_y + self.frame_height:
                return True

        return False

    def get_relative_pos(self, pos):
        x = pos[0] - self.frame_x
        y = pos[1] - self.frame_y
        return x, y
