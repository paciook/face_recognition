import os
import cv2
import numpy as np
from PIL import Image
import pickle


# Get this file and the images directory's path in every os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "imgs")

# Select the face cascade
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# Select the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Declare the needed variables
current_id = 0
label_ids = {}
y_labels = []
x_train = []

# Walk through all images
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			# For every picture:
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
			
			# Join all the labels in a diccionary
			if not label in label_ids:
				label_ids[label] = current_id
				current_id+=1
			id_ = label_ids[label]

			# Open and rezise the image
			raw_gray_image = Image.open(path).convert("L")
			size = (550,550)
			final_image = raw_gray_image.resize(size, Image.ANTIALIAS) # Anti-aliasing filter
			
			# Convert it to a Numpy array
			image_array = np.array(final_image, "uint8")

			# Detect the faces
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				# Captures the region of interest
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

with open("labels.pickle", "wb") as f:
	# Saves the ids
	pickle.dump(label_ids, f)

# Train the recognizer and saves it
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
