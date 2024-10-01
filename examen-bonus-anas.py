import cv2

# Charger le classificateur en cascade pour la détection de visages
cascade_path = r'C:\Users\anas5\Desktop\M2 insa\capteur mvmt\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialiser la capture vidéo à partir de la caméra du PC
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

# Boucle principale de détection en temps réel
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture de la caméra.")
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Analyser chaque visage détecté
    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extraire la région du visage
        face_roi = frame[y:y + h, x:x + w]

        # Appliquer un flou gaussien sur la région du visage
        blurred_face = cv2.GaussianBlur(face_roi, (25, 25), 30)

        # Remplacer la région originale par la région floue
        frame[y:y + h, x:x + w] = blurred_face

    # Afficher l'image avec les visages détectés et floutés
    cv2.imshow("Détection de visages et floutage", frame)

    # Sortir si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
