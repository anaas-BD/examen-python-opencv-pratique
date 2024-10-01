#exercice 4

import cv2

# Charger la vidéo depuis le répertoire
video_path = r'C:\Users\anas5\Desktop\video.mp4'
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo est chargée correctement
if not cap.isOpened():
    print("Erreur : Impossible de charger la vidéo.")
    exit()

# Utiliser le soustracteur de fond MOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialiser le tracker (CSRT dans ce cas)
tracker = None

# Variable pour vérifier si le tracker est initialisé
tracking = False

# Boucle principale de la vidéo
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur de lecture.")
        break

    # Appliquer la soustraction de fond pour obtenir les objets en mouvement
    fgmask = fgbg.apply(frame)
    
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(fgmask, (5, 5), 0)
    
    # Appliquer un seuil pour obtenir une image binaire
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours des objets détectés
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours et sélectionner la plus grande région pour la détection de mouvement
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignorer les petits objets
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Si le tracking est activé, mettre à jour le tracker
    if tracking:
        ok, bbox = tracker.update(frame)
        if ok:
            # Dessiner un rectangle autour de l'objet suivi
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            # Si l'objet est perdu, arrêter le tracking
            tracking = False
            cv2.putText(frame, "Objet perdu", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Afficher l'image
    cv2.imshow("Détection et suivi d'objet", frame)

    # Vérifier si l'utilisateur appuie sur 's' pour sélectionner une région à suivre
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s') and not tracking:
        # Permettre à l'utilisateur de sélectionner manuellement la région d'intérêt
        bbox = cv2.selectROI("Sélection de l'objet à suivre", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Sélection de l'objet à suivre")
        
        # Initialiser le tracker CSRT
        tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frame, bbox)
        
        # Démarrer le suivi si initialisation réussie
        if ok:
            tracking = True

    # Sortir de la boucle si l'utilisateur appuie sur 'q'
    if key == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
