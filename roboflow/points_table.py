import cv2

# Mettez le chemin de votre vidéo ici
video_path = 'ping_pong_match_1.mp4'
points =[]
def afficher_coordonnees(event, x, y, flags, param):
    # Si on fait un CLIC GAUCHE
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        # On dessine un petit cercle rouge pour confirmer le clic
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Cliquez sur les 4 coins", img)

# 1. Ouvrir la vidéo et lire la première image
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
cap.release()

if not ret:
    print("Erreur : Impossible de lire la vidéo")
    exit()

# 2. Afficher l'image et écouter la souris
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

print("Coordonnées sélectionnées :")
for i, point in enumerate(points):
    print(f"Point {i+1}: {point}")
