import numpy as np
import cv2
import pickle

# Select a cascade preset
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# Declare the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# Load the faces labels based on the trained file
labels = {"person_name":1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# Captures the default camera image
cap = cv2.VideoCapture(0)

while(True):
    # Process each frame
    ret, frame = cap.read()

    # Grayscale the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect all faces on the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)

    # Read each face
    for (x, y, w, h) in faces:
        # Declare the region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Try to predict the face
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        # Save the region of interest
        img_item = '6.png'
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)  # BRG (Not RGB)
        stroke = 2

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When done, release and stops
cap.release()
cv2.destroyAllWindows()
