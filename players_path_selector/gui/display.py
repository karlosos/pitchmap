import cv2
import imutils
import pygame
import copy

from .pitch_view import PitchView
from .model_view import ModelView
from .circle import Circle
from .circle import PlayerCircle


class PyGameDisplay:
    def __init__(self, main_window_name, model_window_name, main_object, frame_count):
        self.__main_object = main_object

        self.__window_name = main_window_name
        self.__model_window_name = model_window_name

        self.pitch_model = cv2.imread('data/pitch_model_2.jpg')
        self.__clear_pitch_model = imutils.resize(self.pitch_model, width=600)
        self.pitch_model = copy.copy(self.__clear_pitch_model)

        self.__frame_count = frame_count
        self.__current_frame_id = 0
        self.circles = []

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.__window_name)

        self.__display_surface = pygame.display.set_mode((1380, 815))

        self.__pitch_view_manual = PitchView(x=57, y=5, width=600, height=400)
        self.__pitch_view_automatic = PitchView(x=700, y=5, width=600, height=400)
        self.__model_view_manual = ModelView(x=57, y=410, width=0, height=0)
        self.__model_view_automatic = ModelView(x=700, y=410, width=0, height=0)

        self.current_player = None

    def show(self, frame, frame_number):
        self.__current_frame_id = frame_number
        # background
        self.__display_surface.fill((255, 255, 255))

        # show pitch
        self.__pitch_view_manual.draw(self.__display_surface, frame)
        self.__pitch_view_automatic.draw(self.__display_surface, frame)

        # show model
        self.__model_view_manual.draw(self.__display_surface, self.pitch_model)
        self.__model_view_automatic.draw(self.__display_surface, self.pitch_model)

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

            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.player_circle_event(pos)

        return running

    def player_circle_event(self, pos):
        circle_clicked_flag = False
        for circle in self.circles:
            if circle.is_over(pos):
                self.__main_object.current_player = circle.player
                circle_clicked_flag = True
                self.update_inputs(circle.player)
        if not circle_clicked_flag:
            if self.__model_view_manual.is_over(pos):
                pos = self.__model_view_manual.get_relative_pos(pos)
                self.add_player(pos)
            elif self.__pitch_view_manual.is_over(pos):
                pos = self.__pitch_view_manual.get_relative_pos(pos)
                pos = self.__main_object.player_pos_translated(pos)
                if pos is not None:
                    self.add_player(pos)

    def update(self):
        pygame.display.update()

    def add_player(self, pos):
        player = self.__main_object.add_player(pos)
        self.update_inputs(player)

    def update_inputs(self, player):
        self.current_player = player

    def show_model(self, players_detected, players_automatic):
        self.clear_model()

        circles_detected = []
        circles_manual = []

        pitch_view_circles = []

        for player in players_detected:
            circle = PlayerCircle(player, radius=5, start_x=self.__model_view_automatic.x,
                                  start_y=self.__model_view_automatic.y)
            player_frame_position = self.__main_object.model_to_pitchview(player.position, is_manual=False)
            #circle_pitch_view = Circle(x=player_frame_position[0], y=player_frame_position[1],
            #                           start_x=self.__pitch_view_manual.frame_x, start_y=self.__pitch_view_manual.frame_y,
            #                           radius=2, color=(0, 255, 255))
            #pitch_view_circles.append(circle_pitch_view)
            circle2 = PlayerCircle(player, radius=5, start_x=self.__pitch_view_automatic.frame_x,
                                  start_y=self.__pitch_view_automatic.frame_y, custom_position=player_frame_position)
            circles_detected.append(circle)
            circles_detected.append(circle2)

        for player in players_automatic:
            circle = PlayerCircle(player, radius=5, start_x=self.__model_view_manual.x,
                                  start_y=self.__model_view_manual.y)
            circles_manual.append(circle)
            player_frame_position = self.__main_object.model_to_pitchview(player.position, is_manual=True)
            circle2 = PlayerCircle(player, radius=5, start_x=self.__pitch_view_manual.frame_x,
                                  start_y=self.__pitch_view_manual.frame_y, custom_position=player_frame_position)
            circles_detected.append(circle)
            circles_detected.append(circle2)

        for circle in pitch_view_circles:
            circle.draw(self.__display_surface)

        for circle in circles_detected:
            if circle.player == self.__main_object.current_player:
                circle.highlight()
            else:
                circle.reset_highlight()
            circle.draw(self.__display_surface)

        for circle in circles_manual:
            if circle.player == self.__main_object.current_player:
                circle.highlight()
            else:
                circle.reset_highlight()
            circle.draw(self.__display_surface)

    def clear_model(self):
        self.pitch_model = copy.copy(self.__clear_pitch_model)

    def create_model_window(self):
        pass

    def close_model_window(self):
        pass

    def close_windows(self):
        pygame.display.quit()
