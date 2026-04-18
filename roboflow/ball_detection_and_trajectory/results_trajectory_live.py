import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# --- CONFIGURATION LIVE ---
path = '/Users/philomenecarrel/4A/projet_info_ping_pong/PingPongTracking/roboflow/'
model_path = path + 'models/best-2.pt'

# 0 est généralement l'ID de la webcam intégrée (Mac/PC)
# Si tu as une caméra externe, essaie 1 ou 2.
camera_id = 0 

taille_trace = 15  # On augmente un peu pour le live
model = YOLO(model_path)
points = deque(maxlen=taille_trace)

# 1. Ouvrir le flux caméra au lieu du fichier
cap = cv2.VideoCapture(camera_id)

# Optionnel : Forcer une résolution pour gagner en FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 2. EFFET MIROIR (Indispensable pour le live)
    # Pour que ta gauche soit la gauche de l'écran, on flip horizontalement
    frame = cv2.flip(frame, 1)

    # --- INFÉRENCE ---
    # En live, on peut baisser un peu la conf (ex: 0.5) si la balle bouge trop vite
    results = model(frame, verbose=False, conf=0.5, iou=0.7) 
    result = results[0]
    
    ball_found_this_frame = False

    if result.boxes:
        # On cherche la balle parmi les détections
        for box in result.boxes:
            if box.cls[0] == 0: # ID de la classe balle
                x, y, w, h = box.xywh[0].cpu().numpy()
                center = (int(x), int(y))
                points.append(center)
                cv2.circle(frame, center, 7, (0, 0, 255), -1)
                ball_found_this_frame = True
                break # On ne prend que la "meilleure" balle
    
    # 3. ASTUCE : Si on ne trouve pas la balle, on peut ajouter "None" 
    # pour éviter que la ligne ne relie des points trop éloignés dans le temps
    if not ball_found_this_frame:
        points.append(None)
        
    # --- DESSIN DE LA TRAJECTOIRE ---
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
            
        # Dessin d'une ligne jaune
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)

    # Affichage des FPS (Utile pour vérifier la fluidité en live)
    cv2.putText(frame, "LIVE - Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Tracking Live Ping-Pong", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()