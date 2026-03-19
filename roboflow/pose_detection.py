import cv2
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
model_obj = YOLO('models/best-2.pt') 
model_pose = YOLO('models/yolo11x-pose.pt')

# Paramètres d'affichage du texte d'impact
DUREE_AFFICHAGE = 1.5 
affichage_expiration = 0
dernier_coup = ""

def boxes_intersect(box_ball, box_racket):
    return not (box_ball[2] < box_racket[0] or 
                box_ball[0] > box_racket[2] or 
                box_ball[3] < box_racket[1] or 
                box_ball[1] > box_racket[3])

def classifier_coup(keypoints):
    kp = keypoints.xy[0].cpu().numpy()
    if len(kp) < 11: return "Inconnu"
    # Logique Poignet (10) vs Coude (8)
    return "COUP DROIT" if kp[10][0] < kp[8][0] else "REVERS"

# --- BOUCLE VIDÉO ---
cap = cv2.VideoCapture("dataset_labelise/video_simple/video_simple.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    temps_actuel = time.time()

    # 1. Détection de la Balle et de la Raquette (YOLO Object)
    results_obj = model_obj(frame, verbose=False, conf=0.3)[0]
    
    # Variables pour stocker les coordonnées pour la logique d'impact
    ball_box_for_logic = None
    racket_box_for_logic = None

    # --- NOUVEAU : Dessiner TOUTES les détections à chaque frame ---
    for box in results_obj.boxes:
        cls = int(box.cls[0])
        coords = box.xyxy[0].cpu().numpy().astype(int) # Coordonnées [x1, y1, x2, y2]
        
        if cls == 0: # C'est la BALLE
            ball_box_for_logic = coords # Stocker pour l'impact plus tard
            # Dessiner en BLEU
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Balle", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        elif cls == 1: # C'est la RAQUETTE
            racket_box_for_logic = coords # Stocker pour l'impact plus tard
            # Dessiner en ROUGE
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            cv2.putText(frame, "Raquette", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 2. Détection de l'impact (Logique uniquement, plus de dessin ici)
    if ball_box_for_logic is not None and racket_box_for_logic is not None:
        if boxes_intersect(ball_box_for_logic, racket_box_for_logic):
            # L'impact a lieu : on lance la pose
            results_pose = model_pose(frame, verbose=False)[0]
            
            if results_pose.keypoints:
                dernier_coup = classifier_coup(results_pose.keypoints)
                # On planifie l'affichage du texte pour plus tard
                affichage_expiration = temps_actuel + DUREE_AFFICHAGE

    # 3. AFFICHAGE PERSISTANT DU TEXTE (Si impact récent)
    if temps_actuel < affichage_expiration:
        # Texte en VERT
        cv2.putText(frame, f"IMPACT : {dernier_coup}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Arbitrage Automatique", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()