import cv2
import numpy as np
import matplotlib.pyplot as plt


def test_selectivite_couleur(image_path):

    frame = cv2.imread(image_path)
    if frame is None:
        print("Erreur : Image non trouvée.")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filtre ORANGE
    # Teinte env 10-25, Saturation forte (>120), Luminosité forte (>100)
    lower_orange = np.array([10, 120, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Filtre BLANC
    # Teinte peu importe (0-180), Saturation très faible (<40), Luminosité très forte (>200)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)


    plt.figure(figsize=(15, 5))
    plt.suptitle("Les résultats du traitement en filtre de Computer Vision ", fontsize=16)

    # Image Originale
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Image Originale")
    plt.axis('off')

    # Vision du filtre ORANGE
    plt.subplot(1, 3, 2)
    plt.imshow(mask_orange, cmap='gray')
    plt.title("Filtre ORANGE\n(Signal propre = Tracking facile)")
    plt.axis('off')

    # Vision du filtre BLANC
    plt.subplot(1, 3, 3)
    plt.imshow(mask_white, cmap='gray')
    plt.title("Filtre BLANC\n(Trop de bruit = Tracking plus difficile)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

test_selectivite_couleur("data/videos/balle_orange_2_crop.png")
test_selectivite_couleur("data/videos/balle_blanche_crop.png")