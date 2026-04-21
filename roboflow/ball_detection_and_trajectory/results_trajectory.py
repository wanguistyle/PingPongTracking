import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

path='/Users/philomenecarrel/4A/projet_info_ping_pong/PingPongTracking/roboflow/'

model_path = path + 'models/best-2.pt'  
video_path = path + 'dataset_labelise/Sombre_echange/sombre_echanges - Trim.mp4'   
taille_trace = 10    

model = YOLO(model_path)

points = deque(maxlen=taille_trace)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model(frame, verbose=False, conf=0.8, iou=0.8) 
    result = results[0]
    
    # Si une balle est détectée
    if result.boxes:
        for box in result.boxes:
            if box.cls[0] == 0:
                # On prend la boîte avec la plus grande confiance
                box = result.boxes[0]
                x, y, w, h = box.xywh[0].cpu().numpy()
                center = (int(x), int(y))
                points.append(center)
                
                # Dessiner la balle actuelle
                cv2.circle(frame, center, 5, (0, 0, 255), -1) 

    for i in range(1, len(points)):
        # Si un des points est manquant on saute
        if points[i - 1] is None or points[i] is None:
            continue
        thickness = int(np.sqrt(taille_trace / float(i + 1)) * 2.5)
        
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2) 

    # Affichage
    cv2.imshow("Trajectoire Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()