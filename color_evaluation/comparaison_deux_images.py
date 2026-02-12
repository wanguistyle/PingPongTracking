import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def comparer_deux_images(image_path_1, image_path_2, titre1="Image 1 (Balle Orange)", titre2="Image 2 (Balle Blanche)"):
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)

    if img1 is None or img2 is None:
        print("Erreur : Impossible de lire l'une des images. Vérifiez les chemins.")
        return

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Comparaison Spectrale : Pourquoi la couleur Orange est meilleure', fontsize=16)

    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(titre1)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(titre2)
    axes[0, 1].axis('off')


    hist_h1 = cv2.calcHist([hsv1], [0], None, [180], [0, 180])
    hist_h2 = cv2.calcHist([hsv2], [0], None, [180], [0, 180])


    axes[1, 0].plot(hist_h1, color='orange', linewidth=2)
    axes[1, 0].axvspan(10, 25, color='orange', alpha=0.3, label='Zone Orange')
    axes[1, 0].set_title("Spectre TEINTE (H)")
    axes[1, 0].set_ylabel("Pixels")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)


    axes[1, 1].plot(hist_h2, color='gray', linewidth=2)
    axes[1, 1].axvspan(10, 25, color='orange', alpha=0.1, label='Zone Orange') # Juste pour comparer
    axes[1, 1].set_title("Spectre TEINTE (H)")
    axes[1, 1].grid(alpha=0.3)

    hist_s1 = cv2.calcHist([hsv1], [1], None, [256], [0, 256])
    hist_s2 = cv2.calcHist([hsv2], [1], None, [256], [0, 256])

    axes[2, 0].plot(hist_s1, color='purple', linewidth=2)
    axes[2, 0].set_title("Spectre SATURATION (S)")
    axes[2, 0].set_xlabel("0 = Gris/Blanc  --->  255 = Couleur Vive")
    axes[2, 0].set_ylabel("Pixels")
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(hist_s2, color='purple', linewidth=2)
    axes[2, 1].set_title("Spectre SATURATION (S)")
    axes[2, 1].set_xlabel("0 = Gris/Blanc  --->  255 = Couleur Vive")
    axes[2, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

comparer_deux_images("data/videos/balle_orange_2_crop.png", "data/videos/balle_blanche_crop.png")
