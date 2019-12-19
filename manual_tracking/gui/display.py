import cv2
import imutils
import pygame
import copy

from .pitch_view import PitchView
from .model_view import ModelView
from .circle import PlayerCircle
from .circle import LastPlayerCircle
from .circle import CalibrationCircle
from .input_box import InputBox
from .button import Button


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
        self.calibration_circles = []

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
        self.__button_calibration = Button(x=900, y=100, width=75, height=32, text="Calibrate")
        self.__button_transformation = Button(x=900, y=150, width=75, height=32, text="Transform")

        self.calibration_state = False

    def show(self, frame, frame_transformed, frame_number):
        self.__current_frame_id = frame_number
        # background
        self.__display_surface.fill((255, 255, 255))

        # show pitch
        self.__pitch_view.draw(self.__display_surface, frame)

        # show transformed view
        self.__transform_view.draw(self.__display_surface, frame_transformed)

        # show model
        self.__model_view.draw(self.__display_surface, self.pitch_model)

        # buttons
        self.__button_update.draw(self.__display_surface)
        self.__button_delete.draw(self.__display_surface)
        self.__button_calibration.draw(self.__display_surface)
        self.__button_transformation.draw(self.__display_surface)

        # show input boxes
        for box in self.__input_boxes:
            box.draw(self.__display_surface)

        if self.calibration_state:
            for circle in self.calibration_circles:
                circle.draw(self.__display_surface)

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
                if self.calibration_state:
                    if event.button == 1:
                        if self.__pitch_view.is_over(pos):
                            relative_pos = self.__pitch_view.get_relative_pos(pos)
                            self.add_point_main_window(pos, relative_pos)
                        elif self.__model_view.is_over(pos):
                            relative_pos = self.__model_view.get_relative_pos(pos)
                            self.add_point_model_window(pos, relative_pos)
                else:
                    self.player_circle_event(pos)

                if self.__button_update.is_over(pos):
                    player_id = int(self.__input_box_player_id.text)
                    player_color = int(self.__input_box_player_color.text)
                    self.__main_object.change_player_id(player_id)
                    self.__main_object.change_player_color(player_color)
                elif self.__button_delete.is_over(pos):
                    self.__main_object.delete_player()
                elif self.__button_calibration.is_over(pos):
                    self.calibration_state = self.__main_object.calibration()
                elif self.__button_transformation.is_over(pos):
                    self.__main_object.find_homography()
                for circle in self.calibration_circles:
                    if circle.is_over(pos) and event.button == 3: # right click
                        circle.hit = True
                    elif circle.is_over(pos) and event.button == 2: # middle click
                        self.remove_calibration_point(circle)

            elif event.type == pygame.MOUSEBUTTONUP:
                for circle in self.calibration_circles:
                    circle.hit = False
            elif event.type == pygame.MOUSEMOTION:
                self.__button_update.update_hover(pos)
                self.__button_delete.update_hover(pos)
                self.__button_calibration.update_hover(pos)
                self.__button_transformation.update_hover(pos)

            for input_box in self.__input_boxes:
                input_box.handle_event(event)

            for circle in self.calibration_circles:
                if circle.hit:
                    circle.move()
                    print(self.__main_object.calibrator.points)

        return running

    def player_circle_event(self, pos):
        circle_clicked_flag = False
        for circle in self.circles:
            if circle.is_over(pos):
                self.__main_object.current_player = circle.player
                circle_clicked_flag = True
                self.update_inputs(circle.player)
        if not circle_clicked_flag and self.__model_view.is_over(pos):
            pos = self.__model_view.get_relative_pos(pos)
            self.add_player(pos)

    def update(self):
        self.__button_update.update()
        self.__button_delete.update()
        self.__button_calibration.update(self.calibration_state)
        self.__button_transformation.update()

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
        self.pitch_model = copy.copy(self.__clear_pitch_model)

    def create_model_window(self):
        pass

    def close_model_window(self):
        pass

    def close_windows(self):
        pygame.display.quit()

    def add_point_main_window(self, pos, relative_pos):
        if self.__main_object.calibrator.enabled:
            index, point = self.__main_object.calibrator.add_point_main_window(relative_pos)
            if index:
                circle = CalibrationCircle(index=index, pos=pos, point=point, point_index=0)
                self.calibration_circles.append(circle)

    def add_point_model_window(self, pos, relative_pos):
        index, point = self.__main_object.calibrator.add_point_model_window(relative_pos)
        if index:
            circle = CalibrationCircle(index=index, pos=pos, point=point, point_index=1)
            self.calibration_circles.append(circle)

    def remove_calibration_point(self, circle):
        circle.point[circle.point_index] = None
        self.__main_object.calibrator.clean_calibration_points()
        self.refresh_points()

    def refresh_points(self):
        self.calibration_circles = []
        points = self.__main_object.calibrator.points
        for key, point in points.items():
            pitch_view_point = point[0]
            model_view_point = point[1]

            pitch_view_circle = CalibrationCircle(index=key, pos=pitch_view_point, point=point, point_index=0)
            pitch_view_circle.x = pitch_view_circle.x + self.__pitch_view.frame_x
            pitch_view_circle.y = pitch_view_circle.y + self.__pitch_view.frame_y
            self.calibration_circles.append(pitch_view_circle)

            model_view_circle = CalibrationCircle(index=key, pos=model_view_point, point=point, point_index=0)
            model_view_circle.x = model_view_circle.x + 740
            model_view_circle.y = model_view_circle.y + 210
            self.calibration_circles.append(model_view_circle)
