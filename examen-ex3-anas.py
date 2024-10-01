#exercice 3

import cv2
import numpy as np

# Charger les modèles pré-entrainés
face_net = cv2.dnn.readNetFromCaffe(
    'C:\\Users\\anas5\\Desktop\\M2 insa\\capteur mvmt\\deploy.prototxt.txt',
    'C:\\Users\\anas5\\Desktop\\M2 insa\\capteur mvmt\\res10_300x300_ssd_iter_140000.caffemodel'
)

age_net = cv2.dnn.readNetFromCaffe(
    'C:\\Users\\anas5\\Desktop\\M2 insa\\capteur mvmt\\age_deploy.prototxt',
    'C:\\Users\\anas5\\Desktop\\M2 insa\\capteur mvmt\\age_net.caffemodel'
)

gender_net = cv2.dnn.readNetFromCaffe(
    'C:\\Users\\anas5\\Desktop\\M2 insa\\capteur mvmt\\gender_deploy.prototxt',
    'C:\\Users\\anas5\\Desktop\\M2 insa\\capteur mvmt\\gender_net.caffemodel'
)

# Catégories d'âge et de genre
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Homme', 'Femme']

# Charger l'image ou ouvrir une caméra
image = cv2.imread(r'C:\Users\anas5\Desktop\depp.jpg')  # Remplacez par 0 pour la webcam

# Obtenir les dimensions de l'image
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Détection des visages
face_net.setInput(blob)
detections = face_net.forward()

# Pour chaque détection, évaluer le genre et l'âge
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        # Obtenir les coordonnées du rectangle du visage détecté
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Extraire la région du visage
        face = image[startY:endY, startX:endX]
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # Prétraitement pour la prédiction de genre
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426337, 87.768914, 114.895847), swapRB=False)

        # Prédire le genre
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Prédire l'âge
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Dessiner le rectangle autour du visage et afficher le genre et l'âge
        label = f"{gender}, {age}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Afficher l'image résultante
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
