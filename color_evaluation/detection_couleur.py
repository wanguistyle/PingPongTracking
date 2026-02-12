import cv2
import numpy as np

def draw_histogram(frame, width=400, height=300):
    """
    Crée une image contenant l'histogramme de la TEINTE (Hue)
    """
    # Conversion en HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    # Normalisation pour que le pic le plus haut touche le haut de l'image
    cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
    
    # Création de l'image de fond pour l'histogramme (fond noir)
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # On dessine les barres
    bin_w = int(round(width / 180))
    
    for i in range(180):

        color_hsv = np.array([[[i, 255, 255]]], dtype=np.uint8)
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
        
        cv2.line(hist_img, 
                 (i * bin_w, height), 
                 (i * bin_w, height - int(hist[i])), 
                 color_rgb, 
                 bin_w)
                 
    # Zone orange
    cv2.rectangle(hist_img, (10*bin_w, 0), (25*bin_w, height), (255, 255, 255), 1)
    cv2.putText(hist_img, "Zone Orange", (10*bin_w, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return hist_img

def analyze_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur d'ouverture vidéo")
        return

    print("Appuyez sur 'q' pour quitter l'analyse.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        
        frame = cv2.resize(frame, (600, 400))
        
        # Générer l'image de l'histogramme
        hist_view = draw_histogram(frame, width=600, height=200)
        
        # Empiler verticalement : Vidéo en haut, Spectre en bas
        combined_view = np.vstack((frame, hist_view))
        
        cv2.imshow('Analyseur Spectral Temps Reel', combined_view)
        
        # Ralentir un peu pour voir l'effet
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

analyze_video_stream("data/videos/philomene_avec_sans_balle.mov")