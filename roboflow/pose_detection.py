from ultralytics import YOLO
# --- CONFIGURATION ---
model = YOLO('yolo11x-pose.pt')        # Le chemin vers votre modèle entraîné
video_path = '/Users/philomenecarrel/4A/projet_info_ping_pong/PingPongTracking/roboflow/datas/ping_pong_match_1.mp4'   # Votre vidéo de match
# --------------------- 
results = model(video_path, verbose=False, show=True,vid_stride=10)