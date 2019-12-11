from manual_tracking.frame import loader
from manual_tracking.gui import display

import imutils


class ManualTracker:
    def __init__(self):
        self.video_name = 'baltyk_starogard_1.mp4'
        self.__window_name = f'PitchMap: {self.video_name}'

        self.fl = loader.FrameLoader(self.video_name)
        self.__display = display.PyGameDisplay(main_window_name=self.__window_name, model_window_name="2D Pitch Model",
                                               main_object=self, frame_count=self.fl.get_frames_count())
        self.out_frame = None

    def load_next_frame(self):
        frame = self.fl.load_frame()
        try:
            frame = imutils.resize(frame, width=600)
        except AttributeError:
            return
        self.out_frame = frame

    def load_previous_frame(self):
        frame_number = self.fl.get_current_frame_position()
        self.fl.set_current_frame_position(frame_number-2)
        self.load_next_frame()

    def loop(self):
        self.load_next_frame()
        while True:
            is_exit = not self.__display.input_events()
            if is_exit:
                break

            frame_number = self.fl.get_current_frame_position()
            self.__display.show_model()
            self.__display.show(self.out_frame, frame_number)
            self.__display.update()


if __name__ == '__main__':
    mt = ManualTracker()
    mt.loop()
