import cv2
import imutils
import pygame
import copy

from .pitch_view import PitchView
from .model_view import ModelView
from .circle import PlayerCircle
from .circle import LastPlayerCircle
from .input_box import InputBox
from .button import Button


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
        self.circles = []

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.__window_name)

        self.__display_surface = pygame.display.set_mode((1380, 815))

        self.__pitch_view = PitchView(x=57, y=5, width=600, height=400)
        self.__transform_view = PitchView(x=57, y=410, width=600, height=400)
        self.__model_view = ModelView(x=740, y=210, width=0, height=0)

        self.__input_box_player_id = InputBox(740, 100, 50, 32)
        self.__input_box_player_color = InputBox(740, 150, 50, 32)
        self.__input_boxes = [self.__input_box_player_id, self.__input_box_player_color]

        self.__button_update = Button(x=800, y=100, width=50, height=32, text="Update")
        self.__button_delete = Button(x=800, y=150, width=50, height=32, text="Delete")

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

        self.__button_update.draw(self.__display_surface)
        self.__button_delete.draw(self.__display_surface)

        # show input boxes
        for box in self.__input_boxes:
            box.draw(self.__display_surface)

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
                circle_clicked_flag = False
                for circle in self.circles:
                    if circle.is_over(pos):
                        self.__main_object.current_player = circle.player
                        circle_clicked_flag = True
                        self.update_inputs(circle.player)
                if not circle_clicked_flag and self.__model_view.is_over(pos):
                    pos = self.__model_view.get_relative_pos(pos)
                    self.add_player(pos)
                if self.__button_update.is_over(pos):
                    player_id = int(self.__input_box_player_id.text)
                    player_color = int(self.__input_box_player_color.text)
                    self.__main_object.change_player_id(player_id)
                    self.__main_object.change_player_color(player_color)
                if self.__button_delete.is_over(pos):
                    self.__main_object.delete_player()

            for input_box in self.__input_boxes:
                input_box.handle_event(event)

        return running

    def update(self):
        for box in self.__input_boxes:
            box.update()

        pygame.display.update()

    def add_player(self, pos):
        player = self.__main_object.add_player(pos)
        self.update_inputs(player)

    def update_inputs(self, player):
        self.__input_box_player_id.text = str(player.id)
        self.__input_box_player_color.text = str(player.color)

    def show_model(self, players, last_frame_players):
        self.clear_model()
        self.circles = []

        old_circles = []
        for player in last_frame_players:
            circle = LastPlayerCircle(player, radius=5)
            old_circles.append(circle)
        for circle in old_circles:
            circle.draw(self.__display_surface)

        circles = self.circles
        for player in players:
            circle = PlayerCircle(player, radius=5)
            circles.append(circle)

        for circle in circles:
            if circle.player == self.__main_object.current_player:
                circle.highlight()
            else:
                circle.reset_highlight()
            circle.draw(self.__display_surface)

    def clear_model(self):
        self.__pitch_model = copy.copy(self.__clear_pitch_model)

    def create_model_window(self):
        pass

    def close_model_window(self):
        pass

    def close_windows(self):
        pygame.display.quit()
