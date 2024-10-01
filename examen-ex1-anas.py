#exercice 1

import cv2
import numpy as np


# Charger l'image en utilisant OpenCV
image = cv2.imread(r'C:\Users\anas5\Desktop\depp.jpg')

# Vérifier si l'image a été correctement chargée
if image is None:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")
    exit()

# Convertir l'image en niveaux de gris
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer le filtre Sobel pour détecter les contours horizontaux et verticaux
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Contours dans la direction horizontale (axe x)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Contours dans la direction verticale (axe y)

# Convertir les résultats en format approprié pour l'affichage (normalisation)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combiner les deux images de contours pour obtenir une image finale
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Afficher les images avec OpenCV
cv2.imshow('Image Grayscale', gray_image)
cv2.imshow('Contours Sobel X', sobel_x)
cv2.imshow('Contours Sobel Y', sobel_y)
cv2.imshow('Contours Sobel Combinés', sobel_combined)

# Attendre que l'utilisateur appuie sur une touche pour fermer les fenêtres
cv2.waitKey(0)
cv2.destroyAllWindows()





# Charger l'image
image = cv2.imread(r'C:\Users\anas5\Desktop\depp.jpg', cv2.IMREAD_GRAYSCALE)

# Vérifier si l'image a été correctement chargée
if image is None:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")
    exit()

### Partie 1 : Transformation de Fourier

# Appliquer la transformation de Fourier
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Calculer le spectre de fréquence (magnitude)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

# Manipuler le spectre - Pour simplifier, nous appliquons une suppression de basse fréquence
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
# Créer un masque avec une zone centrale noire (suppression des basses fréquences)
mask = np.ones((rows, cols, 2), np.uint8)
r = 30  # Rayon du cercle
mask[crow-r:crow+r, ccol-r:ccol+r] = 0

# Appliquer le masque et la transformation inverse
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Afficher les résultats avec OpenCV
cv2.imshow("Spectre de fréquence", np.uint8(magnitude_spectrum))
cv2.imshow("Image après transformation inverse", cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

### Partie 2 : Segmentation par Seuillage Adaptatif

# Appliquer un seuillage adaptatif pour segmenter l'image
thresh_adaptatif = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)

# Afficher l'image segmentée par seuillage adaptatif
cv2.imshow("Segmentation par seuillage adaptatif", thresh_adaptatif)

# Attendre qu'une touche soit pressée et fermer toutes les fenêtres
cv2.waitKey(0)
cv2.destroyAllWindows()