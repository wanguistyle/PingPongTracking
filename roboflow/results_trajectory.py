import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

path='/Users/philomenecarrel/4A/projet_info_ping_pong/PingPongTracking/roboflow/'
# --- CONFIGURATION ---
model_path = path + 'models/best-2.pt'        # Le chemin vers votre modèle entraîné
video_path = path + 'dataset_labelise/Sombre_echange/sombre_echanges - Trim.mp4'   # Votre vidéo de match
taille_trace = 10           # Nombre de points à garder (longueur de la queue)
# ---------------------

# 1. Charger le modèle
model = YOLO(model_path)

# 2. Préparer la mémoire (la "queue" de la comète)
# deque est une liste optimisée. Si on ajoute un 31ème point, le 1er disparaît automatiquement.
points = deque(maxlen=taille_trace)

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- INFERENCE ---
    results = model(frame, verbose=False, conf=0.8, iou=0.8) # conf est le seuil de confiance (optionnel)
    result = results[0]
    
    # Si une balle est détectée
    if result.boxes:
        for box in result.boxes:
            if box.cls[0] == 0:
                # On prend la boîte avec la plus grande confiance (la première)
                box = result.boxes[0]
                x, y, w, h = box.xywh[0].cpu().numpy()
                center = (int(x), int(y))
                
                # AJOUTER LE POINT À LA MÉMOIRE
                points.append(center)
                
                # Dessiner la balle actuelle
                cv2.circle(frame, center, 5, (0, 0, 255), -1) # Point Rouge
        
    # --        - DESSIN DE LA TRAJECTOIRE ---
    # On         parcourt tous les points mémorisés pour les relier entre eux
    for i in range(1, len(points)):
        # Si un des points est manquant (ex: balle perdue de vue), on saute
        if points[i - 1] is None or points[i] is None:
            continue
            
        # Dessiner une ligne entre le point précédent (i-1) et le point actuel (i)
        # Épaisseur variable (optionnel) : plus c'est vieux, plus c'est fin
        thickness = int(np.sqrt(taille_trace / float(i + 1)) * 2.5)
        
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2) # Ligne Jaune

    # Affichage
    cv2.imshow("Trajectoire Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()