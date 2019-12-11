import cv2
import imutils
import pygame
import copy

from .pitch_view import PitchView
from .model_view import ModelView


class PyGameDisplay:
    def __init__(self, main_window_name, model_window_name, main_object, frame_count):
        self.__main_object = main_object

        self.__window_name = main_window_name
        self.__model_window_name = model_window_name

        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__clear_pitch_model = imutils.resize(self.__pitch_model, width=600)
        self.__pitch_model = copy.copy(self.__clear_pitch_model)

        self.__frame_count = frame_count
        self.__current_frame_id = 0

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.__window_name)

        self.__display_surface = pygame.display.set_mode((1380, 815))

        self.__pitch_view = PitchView(x=57, y=5, width=600, height=400)
        self.__transform_view = PitchView(x=57, y=410, width=600, height=400)
        self.__model_view = ModelView(x=740, y=210, width=0, height=0)

    def show(self, frame, frame_number):
        self.__current_frame_id = frame_number
        # background
        self.__display_surface.fill((255, 255, 255))

        # show pitch
        self.__pitch_view.draw(self.__display_surface, frame)

        # show transformed view
        self.__transform_view.draw(self.__display_surface, frame)

        # show model
        self.__model_view.draw(self.__display_surface, self.__pitch_model)

    def input_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    self.__main_object.load_next_frame()
                elif event.key == pygame.K_LEFT:
                    self.__main_object.load_previous_frame()

        return running

    def update(self):
        pygame.display.update()

    def show_model(self):
        pass

    def clear_model(self):
        self.__pitch_model = copy.copy(self.__clear_pitch_model)

    def create_model_window(self):
        pass

    def close_model_window(self):
        pass

    def close_windows(self):
        pygame.display.quit()
