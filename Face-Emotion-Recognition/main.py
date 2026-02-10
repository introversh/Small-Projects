import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

# Emotion labels
emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop the face
        face_img = gray[y:y+h, x:x+w]

        # Preprocess for model
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (1, 48, 48, 1))

        # Predict emotion
        pred = model.predict(face_img)
        emotion_label = emotions[np.argmax(pred)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam frame
    cv2.imshow("Real-Time Emotion Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
