import pygame
import cv2


class ModelView:
    def __init__(self, x=740, y=237, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, win, model):
        frame_size = model.shape[1::-1]
        self.height = frame_size[1]
        self.width = frame_size[0]
        rgb_frame = cv2.cvtColor(model, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.image.frombuffer(rgb_frame, frame_size, 'RGB')
        win.blit(pygame_frame, (self.x, self.y))

    def is_over(self, pos):
        if self.x < pos[0] < self.x + self.width:
            if self.y < pos[1] < self.y + self.height:
                return True

        return False

    def get_relative_pos(self, pos):
        x = pos[0] - self.x
        y = pos[1] - self.y
        return x, y
