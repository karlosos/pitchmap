"""
Display used for displaying images in system window
"""
import video_position_trackbar

import cv2
import imutils

import pygame


class Display:
    def __init__(self, main_window_name, model_window_name, pitchmap, frame_count):
        self.__pitchmap = pitchmap

        self.__window_name = main_window_name
        self.__model_window_name = model_window_name

        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

        self.__frame_count = frame_count

        self.__video_position_trackbar = video_position_trackbar.VideoPositionTrackbar(self.__frame_count,
                                                                                       self.__pitchmap.fl)

        cv2.namedWindow(self.__window_name)
        self.__video_position_trackbar.show_trackbar(0, self.__window_name)
        cv2.setMouseCallback(self.__window_name, self.add_point_main_window)

    def show(self, frame, frame_number):
        cv2.imshow(self.__window_name, frame)
        self.__video_position_trackbar.set_trackbar(frame_number, self.__window_name)

    def show_model(self):
        cv2.imshow(self.__model_window_name, self.__pitch_model)

    def clear_model(self):
        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

    def create_model_window(self):
        cv2.namedWindow(self.__model_window_name)
        cv2.setMouseCallback(self.__model_window_name, self.add_point_model_window)

    def close_model_window(self):
        cv2.destroyWindow(self.__model_window_name)

    @staticmethod
    def close_windows():
        cv2.destroyAllWindows()

    def add_point_model_window(self, event, x, y, flags, params):
        """
        Mouse callback. For adding points of interest on model window for perspective transformation.
        """

        if event == cv2.EVENT_LBUTTONUP:
            index = self.__pitchmap.calibrator.add_point_model_window((x, y))
            if index:
                cv2.circle(self.__pitch_model, (x, y), 3, (0, 255, 0), 5)
                cv2.putText(self.__pitch_model, str(index), (x + 3, y + 3),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    def add_point_main_window(self, event, x, y, flags, params):
        """
        Mouse callback. For adding points of interest for perspective transformation.
        """
        if self.__pitchmap.calibrator.enabled:
            if event == cv2.EVENT_LBUTTONUP:
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


class button():
    def __init__(self, color, x, y, width, height, text=''):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self, win, outline=None):
        # Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x - 2, self.y - 2, self.width + 4, self.height + 4), 0)

        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height), 0)

        if self.text != '':
            font = pygame.font.SysFont("Verdana", 12)
            text = font.render(self.text, 1, (0, 0, 0))
            win.blit(text, (
            self.x + (self.width / 2 - text.get_width() / 2), self.y + (self.height / 2 - text.get_height() / 2)))

    def isOver(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True

        return False

class PyGameDisplay:
    def __init__(self, main_window_name, model_window_name, pitchmap, frame_count):
        self.__pitchmap = pitchmap

        self.__window_name = main_window_name
        self.__model_window_name = model_window_name

        self.__pitch_model = cv2.imread('data/pitch_model.jpg')
        self.__pitch_model = imutils.resize(self.__pitch_model, width=600)

        self.__frame_count = frame_count

        self.__video_position_trackbar = video_position_trackbar.VideoPositionTrackbar(self.__frame_count,
                                                                                       self.__pitchmap.fl)
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self.__window_name)
        self._isWindowCreated = True
        #cv2.namedWindow(self.__window_name)
        #self.__video_position_trackbar.show_trackbar(0, self.__window_name)
        #cv2.setMouseCallback(self.__window_name, self.add_point_main_window)

        self.__display_surface = pygame.display.set_mode((800, 600))

        self.__button1 = button((255, 0, 0), 400, 400, 200, 50, 'Calibration')

    def show(self, frame, frame_number):
        # showing frame
        frame_size = frame.shape[1::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.image.frombuffer(rgb_frame, frame_size, 'RGB')
        self.__display_surface.blit(pygame_frame, (0, 0))
        pygame.display.flip()

        # showing button
        self.__button1.draw(self.__display_surface)


    def keyboard_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.__button1.isOver(pos):
                    print("Clicked the button")

            if event.type == pygame.MOUSEMOTION:
                if self.__button1.isOver(pos):
                    self.__button1.color = (220, 100, 0)
                else:
                    self.__button1.color = (255, 0, 0)

        return running

    def update(self):
        pygame.display.update()

    def show_model(self):
        pass

    def clear_model(self):
        pass

    def create_model_window(self):
        pass

    def close_model_window(self):
        pass

    def close_windows(self):
        pygame.display.quit()
        self._isWindowCreated = False

    def add_point_model_window(self, event, x, y, flags, params):
        pass

    def add_point_main_window(self, event, x, y, flags, params):
        pass