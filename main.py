import cv2
import csv
import numpy as np
from utils.tracking_utils import BallTracker, PlayerDetector, PingPongUmpire

def load_ground_truth(csv_path):
    """Loads the CSV and creates a tracking dictionary for each event."""
    gt_events = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_events.append({
                    'start': int(row['Start']),
                    'end': int(row['End']),
                    'name': row['Name'],
                    'detected_predictions': []
                })
    except Exception as e:
        print(f"Warning: Could not load CSV. {e}")
    return gt_events

def draw_ui_overlay(img, text, position, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    overlay = img.copy()
    (w, h), _ = cv2.getTextSize(text, font, 1.0, 2)
    cv2.rectangle(overlay, (position[0]-10, position[1]-h-20), (position[0]+w+10, position[1]+10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, position, font, 1.0, color, 2, cv2.LINE_AA)

def main():

    #VIDEO_PATH = 'data/Videos_annotees/Sombre_echange/sombre_echanges - Trim.mp4'
    #CSV_PATH = 'data/dataset_labelise/video_simple/video_simple.mp4.csv' 
    VIDEO_PATH = 'data/dataset_labelise/video_simple/video_simple.mp4'
    CSV_PATH = 'data/dataset_labelise/video_simple/video_simple.mp4.csv' 
    gt_events = load_ground_truth(CSV_PATH)
    false_positives = 0 
    
    # Dictionnaire pour lier la prédiction de l'IA au vocabulaire du CSV
    mapping_noms = {
        "COUP DROIT": "coup_droit",
        "REVERS": "revers",
        "Inconnu": "inconnu"
    }

    win_name = "Ping Pong AI Analyst - Evaluation Mode"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    tracker = BallTracker(window_name=win_name)
    tracker.setup_trackbars()
    umpire = PingPongUmpire()
    detector = PlayerDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    display_msg, marker, timer = "", None, 0

    print("Starting Evaluation... Press 'q' to stop early and see results.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.resize(frame, (1080, int(frame.shape[0] * (1080/frame.shape[1]))))
        frame = detector.process(frame)
        frame, ball, _, is_real = tracker.process(frame)

        if ball and is_real:
            # 3. L'arbitre analyse la trajectoire
            result = umpire.update(ball[0], ball[1])
            
            if result:
                display_msg, marker = result
                timer = 40
                
                if "PADDLE" in display_msg:
                    # Si la balle repart, on demande au détecteur quel est le coup
                    type_coup = detector.classifier_coup()
                    display_msg = f"IMPACT : {type_coup}"
                    
                    # --- COMPARAISON AVEC LE CSV ---
                    coup_formate = mapping_noms.get(type_coup, "inconnu")
                    matched_to_gt = False
                    
                    for gt in gt_events:
                        # Si l'impact détecté se trouve dans la fenêtre de frames du CSV
                        if gt['start'] <= frame_idx <= gt['end']:
                            gt['detected_predictions'].append(coup_formate)
                            matched_to_gt = True
                            break
                    
                    # Si on détecte un coup mais qu'on n'est dans aucune fenêtre du CSV
                    if not matched_to_gt:
                        false_positives += 1

        if timer > 0:
            clr = (0, 255, 0) if "PADDLE" in display_msg else (0, 165, 255)
            if "DOUBLE" in display_msg or "FAULT" in display_msg: clr = (0, 0, 255)
            
            debug_text = f"Frame {frame_idx}: {display_msg}"
            draw_ui_overlay(frame, debug_text, (50, 80), clr)
            
            if marker:
                cv2.drawMarker(frame, marker, clr, cv2.MARKER_TILTED_CROSS, 30, 3)
            timer -= 1

        for i in range(1, len(tracker.pts)):
            if tracker.pts[i-1] and tracker.pts[i]:
                cv2.line(frame, tracker.pts[i-1], tracker.pts[i], (0, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # --- CALCUL DES METRIQUES FINALES ---
    print("\n" + "="*60)
    print("                 RÉSULTATS DE L'ÉVALUATION")
    print("="*60)
    
    vrais_positifs = 0
    erreurs_classe = 0
    
    for gt in gt_events:
        preds = gt['detected_predictions']
        if len(preds) > 0:
            # On vérifie si le coup attendu (CSV) fait partie des prédictions enregistrées
            if gt['name'] in preds:
                vrais_positifs += 1
                print(f"[Frames {gt['start']:03d}-{gt['end']:03d}] {gt['name']:<12} : ✅ SUCCÈS")
            else:
                erreurs_classe += 1
                unique_preds = list(set(preds))
                print(f"[Frames {gt['start']:03d}-{gt['end']:03d}] {gt['name']:<12} : ❌ ERREUR DE CLASSE (Vu: {unique_preds})")
        else:
            print(f"[Frames {gt['start']:03d}-{gt['end']:03d}] {gt['name']:<12} : 🙈 MANQUÉ (Faux Négatif)")

    total_gt = len(gt_events)
    faux_negatifs = total_gt - (vrais_positifs + erreurs_classe)
    
    accuracy = (vrais_positifs / total_gt) * 100 if total_gt > 0 else 0
    
    print("-" * 60)
    print(f"Total des coups annotés (CSV) : {total_gt}")
    print(f"✅ Vrais Positifs (Bon moment, bon coup) : {vrais_positifs} ({accuracy:.1f}%)")
    print(f"❌ Erreurs de classification             : {erreurs_classe}")
    print(f"🙈 Faux Négatifs (Impact raté par l'IA)  : {faux_negatifs}")
    print(f"⚠️ Faux Positifs (Impacts fantômes)      : {false_positives}")
    print("=" * 60 + "\n")

if __name__ == "__main__": 
    main()