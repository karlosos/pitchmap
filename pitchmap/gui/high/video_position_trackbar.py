import cv2


class VideoPositionTrackbar:
    def __init__(self, frame_count, frame_loader):
        self.__frame_count = frame_count
        self.__frame_loader = frame_loader

    def on_trackbar_change(self, frame_pos):
        print(f"Change to {int(frame_pos)}")
        self.__frame_loader.set_current_frame_position(int(frame_pos))

    def show_trackbar(self, frame_pos, window_name):
        print(f"Created trackbar: {int(frame_pos)}")

        trackbar_name = 'Pos:'
        cv2.createTrackbar(trackbar_name, window_name, int(frame_pos), self.__frame_count, self.on_trackbar_change)

    def set_trackbar(self, frame_pos, window_name):
        trackbar_name = 'Pos:'
        cv2.setTrackbarPos(trackbar_name, window_name, int(frame_pos))
