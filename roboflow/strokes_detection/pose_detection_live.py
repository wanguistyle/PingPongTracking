import cv2
import time
from ultralytics import YOLO

model_obj = YOLO('models/best-2.pt') 
model_pose = YOLO('models/yolo11n-pose.pt') 
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
    frame = cv2.flip(frame, 1)
    
    temps_actuel = time.time()

    results_obj = model_obj(frame, verbose=False, conf=0.3)[0]
    
    ball_box = None
    racket_box = None

    for box in results_obj.boxes:
        cls = int(box.cls[0])
        coords = box.xyxy[0].cpu().numpy().astype(int)

        if cls == 0:
            ball_box = coords
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
        elif cls == 1:
            racket_box = coords
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)

    if boxes_intersect(ball_box, racket_box):
        results_pose = model_pose(frame, verbose=False)[0]
        if results_pose.keypoints:
            dernier_coup = classifier_coup(results_pose.keypoints)
            affichage_expiration = temps_actuel + DUREE_AFFICHAGE

    if temps_actuel < affichage_expiration:
        cv2.putText(frame, f"IMPACT : {dernier_coup}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Arbitrage Live", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()