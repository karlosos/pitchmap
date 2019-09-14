```plantuml
!includeurl https://raw.githubusercontent.com/karlosos/RedDress-PlantUML/master/light_theme.puml

title PitchMap class diagram

class PitchMap {
    - video_name
    - window_name
    - fl : FrameLoader
    + calibrator : Calibrator
    - display : Display
    + players[]
    + players_colors[]
    - trackers
    - frame_number
    - tracking_method
    + out_frame

    + loop()
    + tracking(frame)
    + add_tracking_points(frame)
    + cluster_teams()
    + serialize_bounding_boxes(bouding_boxes)
}

class Display {
    - pitchamp : PitchMap
    - window_name
    - model_winow_name
    - pitch_model
    
    + __init__(main_window_name, model_window_name, pitchmap)
    + show()
    + show_model()
    + create_model_window()
    + close_model_window()
    + close_windows()
    + add_point_model_window(event, x, y, flags, params)
    + add_point_main_window(event, x, y, flags, params)
    + add_players_to_model()
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
}

class Calibrator {
    + enabled
    + points{}
    + current_point

    + toggle_enabled
    + add_point_main_window(pos)
    + add_point_model_window(pos)
    + get_points_count()
}

class Mask {
    + grass(frame, for_players=False)
}

class Detect {
    + players_detection(frame)
    + team_detection_for_player(frame, box)
    + edges_detection(img)
    + lines_detection(frame_edges, img)
}

PitchMap --> FrameLoader
PitchMap --> Calibrator
PitchMap --> Display
PitchMap --> Mask
PitchMap --> Detect
Display --> PitchMap
Display --> Calibrator
@enduml
```