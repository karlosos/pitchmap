```plantuml
!includeurl https://raw.githubusercontent.com/karlosos/RedDress-PlantUML/master/light_theme.puml

class PitchMap {
    - video_name
    - window_name
    - fl : FrameLoader
    + calibrator : Calibrator
    - display : Display
    + players[]
    + players_colors[]
    - frame_number
    - tracking_method
    + out_frame
	
	- team_detector
	- tracker

    + loop()
	+ draw_bounding_boxes(frame, grass_mask, bounding_boxes)
	+ start_calibration()
	+ perform_transform()
}

class Display {
    - pitchamp : PitchMap
    - window_name
    - model_winow_name
    - pitch_model
    - video_position_trackbar
    
    + __init__(main_window_name, model_window_name, pitchmap, frame_count)
    + show(frame, frame_number)
    + show_model()
    + create_model_window()
    + close_model_window()
    + close_windows()
    + add_point_model_window(event, x, y, flags, params)
    + add_point_main_window(event, x, y, flags, params)
    + add_players_to_model()
}

class VideoPositionTrackbar {
    - __frame_count
    - __frame_loader
    + on_trackbar_change(frame_pos)
    + show_trackbar(frame_pos, window_name)
}

note right of Display::add_point_model_window
    This is callback from opencv setMouseCallback
end note

class FrameLoader {
    - file_name
    - cap
    - frame_count
    + selected_frames[]

    + __init__(file_name)
    + load_frame()
    + release()
    + select_frames_for_clustering()
    + get_frames_count()
    + set_current_frame_position(frame_idx)
    + get_current_frame_position()
}

class Calibrator {
    + enabled
    + points{}
    + current_point

    + toggle_enabled
    + add_point_main_window(pos)
    + add_point_model_window(pos)
    + get_points_count()
	+ calibrate(frame, players, players_colors)
}

class Mask {
    + grass(frame, for_players=False)
}

class Detect {
    + edges_detection(img)
    + lines_detection(frame_edges, img)
}

class PlayersDetector {
    + detect(frame)
}

class KeyboardActions {
	- input_point(key)
	- input_exit(key)
	- input_transform(key)
	+ key_pressed(key, pitchmap)
}

class Plotting {
	+ plot_colors(colors, labels)
}

class TeamDetection {
	- clf
	- plot: boolean
	
	+ __init__(plot=False)
	+ cluster_teams(selected_frames)
	+ extract_players_colors(frames)
	+ color_detection_for_player(frame, box)
	+ team_detection_for_player(color)
	+ serialize_bounding_boxes(bounding_boxes)
}

class Tracker {
	+ OPENCV_PBJECT_TRACKERS
	- tracking_method
	- trackers
	
	+ __init__(tracking_method)
	+ update(frame)
	+ trakcing(frame)
	+ draw_bounding_boxes(frame, boxes)
	+ add_tracking_points(frame)
	+ is_tracking_enabled()
}

PitchMap --> FrameLoader
PitchMap --> Calibrator
PitchMap --> Display
PitchMap --> Mask
PitchMap --> Detect
PitchMap --> Tracker
PitchMap --> TeamDetection
PitchMap --> KeyboardActions
Display --> PitchMap
Display --> Calibrator
TeamDetection --> Plotting
TeamDetection --> PlayersDetector
TeamDetection --> Mask
Tracker --> PlayersDetector
VideoPositionTrackbar --> FrameLoader
Display --> VideoPositionTrackbar
@enduml
```