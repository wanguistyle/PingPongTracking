import cv2
import os

def extraire_et_sauvegarder_frames(video_path, liste_frames, dossier_sortie):
    """
    video_path : chemin vers la vidéo (ex: 'video_ping_pong.mp4')
    liste_frames : liste d'entiers [10, 45, 120, ...]
    dossier_sortie : nom du dossier où enregistrer les images (ex: 'captures_impacts/')
    """
    # 1. On crée le dossier de sortie s'il n'existe pas
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        print(f"Dossier '{dossier_sortie}' créé avec succès.")
    else:
        print(f"Le dossier '{dossier_sortie}' existe déjà.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    nombre_sauvegardes = 0

    # On trie la liste pour éviter les "aller-retours" inutiles dans la vidéo
    for f_idx in sorted(liste_frames):
        # 2. Positionner le curseur
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        
        # 3. Lire la frame
        ret, frame = cap.read()
        
        if ret:
            # 4. Construire le nom du fichier
            # On utilise .zfill(6) pour avoir des noms comme 'frame_000123.jpg'
            # Cela permet aux fichiers de se trier correctement sur ton ordinateur.
            nom_fichier = f"frame_{str(f_idx).zfill(6)}.jpg"
            chemin_complet = os.path.join(dossier_sortie, nom_fichier)
            
            # 5. ENREGISTRER L'IMAGE (pas d'affichage !)
            # cv2.imwrite gère automatiquement le format .jpg ou .png selon l'extension
            cv2.imwrite(chemin_complet, frame)
            
            print(f"Frame {f_idx} sauvegardée dans {chemin_complet}")
            nombre_sauvegardes += 1
        else:
            print(f"Avertissement : Impossible de lire la frame {f_idx}")

    cap.release()
    print(f"\nTerminé. {nombre_sauvegardes} images ont été enregistrées dans '{dossier_sortie}'.")

# --- EXEMPLE D'UTILISATION ---

# 1. Définir tes paramètres
video_a_analyser = "dataset_labelise/video_simple/video_simple.mp4"
dossier_de_captures = "captures_erreur_haute/"

# 2. Obtenir ta liste de frames (par exemple, tes pics d'erreurs)
# Pour cet exemple, on en prend quelques-unes au hasard
frames_impacts =[60]

# 3. Lancer la sauvegarde
extraire_et_sauvegarder_frames(video_a_analyser, frames_impacts, dossier_de_captures)