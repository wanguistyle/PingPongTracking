import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_frame(frame): 
    # Convertir de BGR (OpenCV) à RGB (Matplotlib)
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.show()

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir {video_path}")
        return None

    # On récupère les propriétés
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Nombre de frames estimées : {frameCount}")
    
    # On utilise une liste dynamique au lieu d'un buffer fixe pour éviter les erreurs d'index
    # si le compte de frames est inexact
    frames = []
    
    while True:
        ret, frame = cap.read()
        
        # Si la lecture échoue (fin de vidéo), on arrête la boucle
        if not ret:
            break
            
        frames.append(frame)

    cap.release()
    
    # Conversion de la liste en tableau NumPy
    video_array = np.array(frames)
    print(f"Vidéo chargée avec succès. Shape finale : {video_array.shape}")
    return video_array

def afficher_histogramme(frame):
    if frame is None:
        print("Erreur : Frame vide fournie à l'histogramme.")
        return

    # OpenCV utilise BGR, donc on définit les couleurs dans cet ordre pour le plot
    couleurs = ('b', 'g', 'r') 
    labels = ('Bleu', 'Vert', 'Rouge')
    
    plt.figure(figsize=(10, 5))
    
    # Affichage de l'image miniature à côté pour référence
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Image analysée")
    plt.axis('off')
    
    # Affichage de l'histogramme
    plt.subplot(1, 2, 2)
    plt.title("Histogramme de couleur (Spectre)")
    plt.xlabel("Intensité (0-255)")
    plt.ylabel("Nombre de Pixels")
    plt.grid(alpha=0.3)
    
    for i, couleur in enumerate(couleurs):
        # Calcule l'histogramme pour le canal i
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        plt.plot(hist, color=couleur, label=labels[i])
        plt.xlim([0, 256])
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Exécution ---

# Chemin vers votre vidéo (Assurez-vous que le chemin est correct)
video_path = "data/videos/philomene_recevoir_balle.mov"

# Vérification que le fichier existe avant de lancer
if os.path.exists(video_path):
    video_data = read_video(video_path)
    
    if video_data is not None and len(video_data) > 0:
        # Afficher l'histogramme de la première image (index 0)
        print("Affichage de l'histogramme pour la première frame...")
        afficher_histogramme(video_data[0])
        
        # Si vous voulez l'histogramme de la balle (souvent orange/blanc), 
        # il faudra peut-être cropper l'image autour de la balle avant.
else:
    print(f"Le fichier n'existe pas : {video_path}")