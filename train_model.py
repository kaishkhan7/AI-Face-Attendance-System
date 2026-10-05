import cv2
import os
import numpy as np

DATASET_PATH = "students"
MODEL_PATH = "encodings/face_model.yml"

faces = []
labels = []

label_id = 0
label_names = {}

for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    label_names[label_id] = person_name

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(label_id)

    label_id += 1


if len(faces) == 0:
    print("Koi face detect nahi hua.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

recognizer.write(MODEL_PATH)

np.save("encodings/label_names.npy", label_names)

print("Training complete!")
print("Students:", label_names)
print("Model saved:", MODEL_PATH)