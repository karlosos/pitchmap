import cv2
import imutils
import pygame
import copy

from .button import Button
from .pitch_view import PitchView
from .model_view import ModelView
from .slider import Slider


class PyGameDisplay:
    def __init__(self, main_window_name, model_window_name, pitchmap, frame_count):
        self.__pitchmap = pitchmap

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

        self.__display_surface = pygame.display.set_mode((1380, 720))

        self.__pitch_view = PitchView()
        self.__model_view = ModelView()
        self.__buttonCalibration = Button(737, 57, 200, 50, 'Calibration')
        self.__buttonTransformation = Button(994, 57, 200, 50, 'Transformation')
        self.__slider = Slider(1, self.__frame_count, 1)

        self.__calibration_state = False
        self.__transformation_state = False

    def show(self, frame, frame_number):
        self.__current_frame_id = frame_number
        # background
        self.__display_surface.fill((255, 255, 255))

        # show pitch
        self.__pitch_view.draw(self.__display_surface, frame)

        # show buttons
        self.__buttonCalibration.draw(self.__display_surface)
        self.__buttonTransformation.draw(self.__display_surface)

        # show model
        self.__model_view.draw(self.__display_surface, self.__pitch_model)

        self.__slider.draw(self.__display_surface)
        self.__slider.set_value(frame_number)

    def input_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    self.__calibration_state = self.__pitchmap.start_calibration()
                elif event.key == pygame.K_t:
                    self.__pitchmap.perform_transform()
                elif event.key == pygame.K_a:
                    self.__pitchmap.accept_transform()
                elif event.key == pygame.K_d:
                    self.__pitchmap.toggle_detecting()
                elif event.key == pygame.K_p:
                    self.__pitchmap.toggle_transforming()

            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.__buttonCalibration.is_over(pos):
                    self.__calibration_state = self.__pitchmap.start_calibration()
                elif self.__buttonTransformation.is_over(pos):
                    self.__pitchmap.perform_transform()
                elif self.__pitch_view.is_over(pos):
                    pos = self.__pitch_view.get_relative_pos(pos)
                    self.add_point_main_window(pos[0], pos[1])
                elif self.__model_view.is_over(pos):
                    pos = self.__model_view.get_relative_pos(pos)
                    self.add_point_model_window(pos[0], pos[1])
                elif self.__slider.button_rect.collidepoint(pos):
                    self.__slider.hit = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.__slider.hit = False
            elif event.type == pygame.MOUSEMOTION:
                if self.__buttonCalibration.is_over(pos):
                    self.__buttonCalibration.is_hover = True
                else:
                    self.__buttonCalibration.is_hover = False

                if self.__buttonTransformation.is_over(pos):
                    self.__buttonTransformation.is_hover = True
                else:
                    self.__buttonTransformation.is_hover = False

        if self.__slider.hit:
            self.__slider.move()
            self.__pitchmap.fl.set_current_frame_position(int(self.__slider.val))
        return running

    def update(self):
        if self.__calibration_state:
            self.__buttonCalibration.state_color = self.__buttonCalibration.COLOR_ENABLED
        elif self.__buttonCalibration.is_hover:
            self.__buttonCalibration.state_color = self.__buttonCalibration.COLOR_HOVER
        else:
            self.__buttonCalibration.state_color = self.__buttonCalibration.COLOR_STANDARD

        self.__buttonCalibration.color = self.__buttonCalibration.state_color

        pygame.display.update()

    def show_model(self, players_2d_positions=None, colors=None):
        if players_2d_positions is None:
            players_2d_positions = []
        if len(players_2d_positions):
            self.add_players_to_model(players_2d_positions, colors)

    def clear_model(self):
        self.__pitch_model = copy.copy(self.__clear_pitch_model)

    def create_model_window(self):
        pass

    def close_model_window(self):
        pass

    def close_windows(self):
        pygame.display.quit()

    def add_point_model_window(self, x, y):
        """
        Mouse callback. For adding points of interest on model window for perspective transformation.
        """

        index = self.__pitchmap.calibrator.add_point_model_window((x, y))
        if index:
            cv2.circle(self.__pitch_model, (x, y), 3, (0, 255, 0), 5)
            cv2.putText(self.__pitch_model, str(index), (x + 3, y + 3),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    def add_point_main_window(self, x, y):
        """
        Mouse callback. For adding points of interest for perspective transformation.
        """
        if self.__pitchmap.calibrator.enabled:
            index = self.__pitchmap.calibrator.add_point_main_window((x, y))
            #print("original, model")
            #print(f"Index: {index}")
            if index:
                cv2.circle(self.__pitchmap.out_frame, (x, y), 3, (0, 255, 0), 5)
                cv2.putText(self.__pitchmap.out_frame, str(index), (x+3, y+3),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    def add_players_to_model(self, players, player_colors):
        self.clear_model()
        for idx, player in enumerate(players):
            cv2.circle(self.__pitch_model, (int(player[0]), int(player[1])), 3, player_colors[idx], 5)
