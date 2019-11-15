import cv2
import imutils
import pygame

from .button import Button


class PyGameDisplay:
    def __init__(self, main_window_name, model_window_name, pitchmap, frame_count):
        self.__pitchmap = pitchmap

        self.__window_name = main_window_name
        self.__model_window_name = model_window_name

        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

        self.__frame_count = frame_count

        # self.__video_position_trackbar = video_position_trackbar.VideoPositionTrackbar(self.__frame_count,
        # self.__pitchmap.fl)
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.__window_name)
        #self.__video_position_trackbar.show_trackbar(0, self.__window_name)

        self.__display_surface = pygame.display.set_mode((1380, 720))

        self.__buttonCalibration = Button(737, 57, 200, 50, 'Calibration')
        self.__buttonTransformation = Button(994, 57, 200, 50, 'Transformation')

        self.__calibration_state = False
        self.__transformation_state = False

    def show(self, frame, frame_number):
        # background
        self.__display_surface.fill((255, 255, 255))
        pygame.draw.rect(self.__display_surface, (0, 0, 0), (57, 57, 600, 600), 0)

        # show pitch
        frame_size = frame.shape[1::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.image.frombuffer(rgb_frame, frame_size, 'RGB')
        self.__display_surface.blit(pygame_frame, (57, 178))

        # show buttons
        self.__buttonCalibration.draw(self.__display_surface)
        self.__buttonTransformation.draw(self.__display_surface)

        # show model
        self.show_model()

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

            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.__buttonCalibration.is_over(pos):
                    self.__calibration_state = self.__pitchmap.start_calibration()
                elif self.__buttonTransformation.is_over(pos):
                    self.__pitchmap.perform_transform()
            elif event.type == pygame.MOUSEMOTION:
                if self.__buttonCalibration.is_over(pos):
                    self.__buttonCalibration.is_hover = True
                else:
                    self.__buttonCalibration.is_hover = False

                if self.__buttonTransformation.is_over(pos):
                    self.__buttonTransformation.is_hover = True
                else:
                    self.__buttonTransformation.is_hover = False

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

    def show_model(self):
        frame_size = self.__pitch_model.shape[1::-1]
        rgb_frame = cv2.cvtColor(self.__pitch_model, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.image.frombuffer(rgb_frame, frame_size, 'RGB')
        self.__display_surface.blit(pygame_frame, (740, 237))
        pass

    def clear_model(self):
        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

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
            print("original, model")
            print(f"Index: {index}")
            if index:
                cv2.circle(self.__pitchmap.out_frame, (x, y), 3, (0, 255, 0), 5)
                cv2.putText(self.__pitchmap.out_frame, str(index), (x+3, y+3),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    def add_players_to_model(self, players, player_colors):
        self.clear_model()
        for idx, player in enumerate(players):
            cv2.circle(self.__pitch_model, (int(player[0]), int(player[1])), 3, player_colors[idx], 5)
