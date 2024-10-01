#exercice 2

import cv2

# Charger l'image en utilisant OpenCV
image = cv2.imread(r'C:\Users\anas5\Desktop\depp.jpg')

# Vérifier si l'image a été correctement chargée
if image is None:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")
    exit()

# Sélectionner une Région d'Intérêt (ROI) - Par exemple, un rectangle autour d'une zone spécifique
x, y, w, h = 100, 100, 200, 200  # Coordonnées de la ROI : coin supérieur gauche (x, y), largeur (w), hauteur (h)
roi = image[y:y+h, x:x+w]

# Appliquer un masque à la ROI (augmenter la luminosité ou changer la couleur)
# Ici, on augmente la luminosité de la ROI
roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=30)  # alpha > 1 augmente le contraste, beta > 0 augmente la luminosité

# Remplacer la région d'origine par la ROI modifiée dans l'image
image[y:y+h, x:x+w] = roi

# Créer une copie de l'image pour appliquer le flou
flou_image = image.copy()

# Appliquer un filtre flou gaussien à l'image complète
flou_image = cv2.GaussianBlur(flou_image, (15, 15), 0)

# Remettre la ROI non floue dans l'image floutée
flou_image[y:y+h, x:x+w] = roi

# Afficher les résultats avec OpenCV
cv2.imshow('Image originale avec ROI modifiée', image)
cv2.imshow('Image finale avec flou en dehors de la ROI', flou_image)

# Attendre qu'une touche soit pressée et fermer les fenêtres
cv2.waitKey(0)
cv2.destroyAllWindows()
