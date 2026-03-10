import cv2
import numpy as np
from utils.tracking_utils_2D import BallTracker, PlayerDetector, TablePnPEstimator, PingPongUmpire, CourtVisualizer

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))

def main(video_path):
    global clicked_points
    VIDEO_PATH = video_path
    WINDOW_MAIN = "Smart Tracker (Video)"
    WINDOW_3D = "3D Court Radar"
    TARGET_WIDTH = 800
    LOOP_VIDEO = True

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_3D, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_MAIN, mouse_callback)

    ball_tracker = BallTracker(buffer_size=32, max_jump_dist=150, window_name=WINDOW_MAIN)
    ball_tracker.setup_trackbars()
    player_detector = PlayerDetector()
    umpire = PingPongUmpire()
    visualizer = CourtVisualizer(scale=0.15)

    stream = cv2.VideoCapture(VIDEO_PATH)
    if not stream.isOpened(): return

    ret, initial_frame = stream.read()
    if not ret: return
    
    scale = TARGET_WIDTH / initial_frame.shape[1]
    height = int(initial_frame.shape[0] * scale)
    initial_frame = cv2.resize(initial_frame, (TARGET_WIDTH, height))
    
    pnp_estimator = TablePnPEstimator(frame_width=TARGET_WIDTH, frame_height=height)
    
    while len(clicked_points) < 4:
        temp_frame = initial_frame.copy()
        for pt in clicked_points:
            cv2.circle(temp_frame, pt, 5, (0, 255, 255), -1)
        
        instruction = f"Click 4 corners: Net L, Near L, Near R, Net R ({len(clicked_points)}/4)"
        cv2.putText(temp_frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WINDOW_MAIN, temp_frame)
        if cv2.waitKey(1) == ord('q'): return

    pnp_estimator.update_camera_pose(clicked_points)
    stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    display_text = ""
    display_timer = 0

    while True:
        ret, frame = stream.read()
        if not ret:
            if LOOP_VIDEO:
                stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ball_tracker.last_center = None
                display_text = ""
                continue
            else: break

        frame = cv2.resize(frame, (TARGET_WIDTH, height))
        
        cv2.polylines(frame, [np.int32(clicked_points)], True, (0, 255, 0), 2)
        for pt in clicked_points:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)

        frame = player_detector.process(frame)
        frame, ball_center, ball_w = ball_tracker.process(frame)
        
        current_table_coords = None
        if ball_center is not None and pnp_estimator.rvec is not None:
            bx, by = ball_center
            
            # Project ball directly to the table plane
            table_coords = pnp_estimator.project_ball_to_table_plane(bx, by)
            
            if table_coords:
                current_table_coords = table_coords
                wx, wy = table_coords
                
                # Umpire now uses the 2D Pixel 'by' to detect the bounce
                bounce_result = umpire.update(by, wx, wy)
                
                if bounce_result:
                    display_text = bounce_result
                    display_timer = 45 

        minimap = visualizer.draw(table_coords=current_table_coords, text=display_text if display_timer > 0 else "")

        if display_timer > 0:
            color = (0, 255, 0) if "IN" in display_text else (0, 0, 255)
            cv2.putText(frame, display_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA)
            display_timer -= 1

        cv2.imshow(WINDOW_MAIN, frame)
        cv2.imshow(WINDOW_3D, minimap)

        if cv2.waitKey(1) == ord('q'): break

    stream.release()
    cv2.destroyAllWindows()
main("roboflow/dataset_labelise/video_lente/video_lente.mp4")