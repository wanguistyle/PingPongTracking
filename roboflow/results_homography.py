import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
model_path = 'yolov11.pt'        # Le chemin vers votre modèle entraîné
video_path = 'ping_pong_match_1.mp4'   # Votre vidéo de match
# ---------------------

# 1. Charger le modèle
model = YOLO(model_path)

# 2. Initialiser la liste pour stocker les points cliqués
points =[]
def afficher_coordonnees(event, x, y, flags, param):
    # Si on fait un CLIC GAUCHE
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        # On dessine un petit cercle rouge pour confirmer le clic
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Cliquez sur les 4 coins", img)

# Ouvrir la vidéo et lire la première image
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
cap.release()

if not ret:
    print("Erreur : Impossible de lire la vidéo")
    exit()

# Afficher l'image et écouter la souris
cv2.imshow("Cliquez sur les 4 coins", img)
cv2.setMouseCallback("Cliquez sur les 4 coins", afficher_coordonnees)

print("--- INSTRUCTIONS ---")
print("Cliquez dans cet ordre précis :")
print("1. Haut-Gauche")
print("2. Haut-Droite")
print("3. Bas-Droite")
print("4. Bas-Gauche")
print("Appuyez sur n'importe quelle touche pour fermer une fois fini.")

cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Définir la transformation de perspective (La "Vue de Dessus")
# Ordre : [Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche]
pts_src = np.float32(points)

# Dimensions réelles (Table de ping-pong standard en cm : 274 x 152.5) -> demi côté de la table et pas largeur entière dans notre cas : 137 x 132 ?
# On peut multiplier par 2 ou 3 pour avoir une image de sortie plus grande (ex: 1 pixel = 1 mm)
scale = 2 
width_real = int(132 * scale)
height_real = int(137 * scale)
pts_dst = np.float32([[0, 0], [width_real, 0], [width_real, height_real], [0, height_real]])

# Création de la matrice magique
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

# Créer une "carte" vide pour dessiner la trajectoire (fond vert ping-pong)
minimap = np.zeros((height_real, width_real, 3), dtype=np.uint8)
minimap[:] = (0, 100, 0) # Couleur verte

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- 3. INFERENCE YOLO ---
    # On lance la détection sur la frame actuelle
    # verbose=False évite de spammer la console
    results = model(frame, verbose=False) 

    # --- 4. RÉCUPÉRATION DES COORDONNÉES ---
    # results est une liste (car on peut traiter plusieurs images à la fois), on prend la première [0]
    result = results[0]
    
    # On regarde s'il y a des boîtes détectées
    if result.boxes:
        for box in result.boxes:
            if box.cls[0] == 0:  # Vérifier que la classe détectée est bien "ball" (classe 0)
            # Récupérer x, y (centre), w, h
            # .cpu().numpy() est important pour convertir le format PyTorch en format utilisable
                x, y, w, h = box.xywh[0].cpu().numpy()
                
                # --- 5. TRANSFORMATION ---
                # On prépare le point pour OpenCV (forme requise : [[[x, y]]])
                ball_point = np.array([[[x, y]]], dtype='float32')
                
                # On applique la matrice
                transformed_point = cv2.perspectiveTransform(ball_point, matrix)
                
                # On récupère les nouvelles coordonnées (u, v) sur la minimap
                u = int(transformed_point[0][0][0])
                v = int(transformed_point[0][0][1])
                
                # --- 6. DESSIN ---
                # Dessiner sur la vidéo originale (cercle rouge sur la balle)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # Dessiner sur la minimap (point blanc)
                # On vérifie que le point est bien "sur la table" avant de dessiner
                if 0 <= u < width_real and 0 <= v < height_real:
                    cv2.circle(minimap, (u, v), 3, (255, 255, 255), -1)

    # Affichage
    cv2.imshow("Video Originale", frame)
    cv2.imshow("Trajectoire Vue de Dessus", minimap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()