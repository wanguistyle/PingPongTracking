import cv2
import time
import pandas as pd
from ultralytics import YOLO

# --- CHARGEMENT DU FICHIER CSV ---
# Assurez-vous d'adapter le chemin vers votre fichier csv
df_annotations = pd.read_csv("/Users/philomenecarrel/4A/projet_info_ping_pong/PingPongTracking/roboflow/dataset_labelise/echange_gymnase/echanges_gymnase2.mp4.csv")

# Variables pour les métriques
metrics = {
    "vrai_positif": 0,         # Coup détecté au bon moment et bonne classe
    "erreur_classe": 0,        # Coup détecté au bon moment mais mauvaise classe
    "faux_positif": 0,         # Coup détecté alors qu'il n'y a rien dans le CSV
    "coups_csv_valides": set() # Pour éviter de compter un même coup plusieurs fois s'il dure plusieurs frames
}

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

# Mapping pour la comparaison avec le CSV
mapping_noms = {
    "COUP DROIT": "coup_droit",
    "REVERS": "revers",
    "Inconnu": "inconnu"
}

# --- BOUCLE VIDÉO ---
cap = cv2.VideoCapture("/Users/philomenecarrel/4A/projet_info_ping_pong/PingPongTracking/roboflow/dataset_labelise/echange_gymnase/echanges_gymnase2.mp4")
frame_idx = 0 # Compteur de frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_idx += 1
    temps_actuel = time.time()

    results_obj = model_obj(frame, verbose=False, conf=0.3)[0]
    
    ball_box_for_logic = None
    racket_box_for_logic = None

    for box in results_obj.boxes:
        cls = int(box.cls[0])
        coords = box.xyxy[0].cpu().numpy().astype(int)
        
        if cls == 0: # C'est la BALLE
            ball_box_for_logic = coords
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Balle", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        elif cls == 1: # C'est la RAQUETTE
            racket_box_for_logic = coords
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            cv2.putText(frame, "Raquette", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Détection de l'impact
    if ball_box_for_logic is not None and racket_box_for_logic is not None:
        if boxes_intersect(ball_box_for_logic, racket_box_for_logic):
            
            results_pose = model_pose(frame, verbose=False)[0]
            if results_pose.keypoints:
                dernier_coup = classifier_coup(results_pose.keypoints)
                affichage_expiration = temps_actuel + DUREE_AFFICHAGE
                
                coup_formate = mapping_noms.get(dernier_coup, "inconnu")

                # --- LOGIQUE DE METRIQUES ---
                # On cherche si la frame actuelle est dans un intervalle du CSV
                match_csv = df_annotations[(df_annotations['Start'] <= frame_idx) & (df_annotations['End'] >= frame_idx)]
                
                if not match_csv.empty:
                    # L'impact se passe bien pendant un intervalle connu
                    index_csv = match_csv.index[0] # Identifiant unique de l'action dans le CSV
                    nom_attendu = match_csv.iloc[0]['Name']
                    
                    # On vérifie si on n'a pas déjà compté cette action précise
                    if index_csv not in metrics["coups_csv_valides"]:
                        if coup_formate == nom_attendu:
                            print(f"[Frame {frame_idx}] Succès : {coup_formate} détecté correctement !")
                            metrics["vrai_positif"] += 1
                        else:
                            print(f"[Frame {frame_idx}] Erreur Classe : Détecté {coup_formate}, attendu {nom_attendu}")
                            metrics["erreur_classe"] += 1
                        
                        # On marque ce coup comme traité pour ne pas le recompter aux frames suivantes
                        metrics["coups_csv_valides"].add(index_csv)
                else:
                    # L'impact se passe EN DEHORS d'un intervalle du CSV
                    print(f"[Frame {frame_idx}] ⚠️ Faux Positif : {coup_formate} détecté hors de tout intervalle !")
                    metrics["faux_positif"] += 1

    if temps_actuel < affichage_expiration:
        cv2.putText(frame, f"IMPACT : {dernier_coup}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Arbitrage Automatique", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- CALCUL ET AFFICHAGE DES METRIQUES FINALES ---
total_coups_reels = len(df_annotations)
faux_negatifs = total_coups_reels - len(metrics["coups_csv_valides"]) # Coups du CSV non détectés

print("\n" + "="*40)
print("📊 RÉSULTATS DES MÉTRIQUES")
print("="*40)
print(f"Total des coups dans le CSV : {total_coups_reels}")
print(f"Vrais Positifs (Bon coup, bon moment) : {metrics['vrai_positif']}")
print(f"Erreurs de classification (Mauvais coup détecté) : {metrics['erreur_classe']}")
print(f"Faux Positifs (Impact fantôme détecté) : {metrics['faux_positif']}")
print(f"Faux Négatifs (Coups ratés/non détectés) : {faux_negatifs}")

if total_coups_reels > 0:
    precision = metrics['vrai_positif'] / (metrics['vrai_positif'] + metrics['faux_positif'] + metrics['erreur_classe'] + 1e-6)
    rappel = metrics['vrai_positif'] / total_coups_reels
    print(f"\nPrécision du système : {precision*100:.2f}%")
    print(f"Rappel du système : {rappel*100:.2f}%")