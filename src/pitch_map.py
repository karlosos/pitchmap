"""
Main file of PitchMap. All process trough loading images from video to displaying 2D map.
"""
import frame_loader
import frame_display
import mask
import detect

import cv2


def exit_user_input():
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        return True
    else:
        return False


def main():
    video_name = 'Dynamic_Barca_Real.mp4'
    fl = frame_loader.FrameLoader(video_name)
    display = frame_display.Display(f'PitchMap: {video_name}')

    while True:
        img = fl.load_frame()
        grass_mask = mask.grass(img)
        bounding_boxes_frame, bounding_boxes, labels = detect.players(grass_mask)

        display.show(bounding_boxes_frame)
        if exit_user_input():
            break

    cv2.destroyAllWindows()
    fl.release()


if __name__ == '__main__':
    main()
