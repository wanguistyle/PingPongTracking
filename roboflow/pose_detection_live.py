import cv2
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
# Utilise des modèles plus légers (n pour Nano) si le live saccade trop
model_obj = YOLO('models/best-2.pt') 
model_pose = YOLO('models/yolo11n-pose.pt') # 'n' est plus rapide que 'x' pour le live

DUREE_AFFICHAGE = 1.5 
affichage_expiration = 0
dernier_coup = ""

def boxes_intersect(box_ball, box_racket):
    if box_ball is None or box_racket is None: return False
    return not (box_ball[2] < box_racket[0] or box_ball[0] > box_racket[2] or 
                box_ball[3] < box_racket[1] or box_ball[1] > box_racket[3])

def classifier_coup(keypoints):
    kp = keypoints.xy[0].cpu().numpy()
    if len(kp) < 11: return "Inconnu"
    return "COUP DROIT" if kp[10][0] < kp[8][0] else "REVERS"

# --- CONFIGURATION LIVE ---
# 0 est généralement l'ID de la webcam intégrée
cap = cv2.VideoCapture(0) 

# Optionnel : Forcer une résolution plus basse pour gagner en FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break

    # Miroir (optionnel) : plus naturel quand on se regarde en live
    frame = cv2.flip(frame, 1)
    
    temps_actuel = time.time()

    # 1. Détection Objets (Balle/Raquette)
    # On augmente un peu la vitesse avec stream=True ou en limitant les classes
    results_obj = model_obj(frame, verbose=False, conf=0.3)[0]
    
    ball_box = None
    racket_box = None

    for box in results_obj.boxes:
        cls = int(box.cls[0])
        coords = box.xyxy[0].cpu().numpy().astype(int)
        
        # Affichage constant
        if cls == 0:
            ball_box = coords
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
        elif cls == 1:
            racket_box = coords
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)

    # 2. Logique d'impact
    if boxes_intersect(ball_box, racket_box):
        results_pose = model_pose(frame, verbose=False)[0]
        if results_pose.keypoints:
            dernier_coup = classifier_coup(results_pose.keypoints)
            affichage_expiration = temps_actuel + DUREE_AFFICHAGE

    # 3. Affichage du texte si impact récent
    if temps_actuel < affichage_expiration:
        cv2.putText(frame, f"IMPACT : {dernier_coup}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Arbitrage Live", frame)
    
    # 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()