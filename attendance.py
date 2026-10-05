import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("encodings/face_model.yml")

# Student names
label_names = np.load(
    "encodings/label_names.npy",
    allow_pickle=True
).item()

# Attendance file
attendance_file = "attendance.csv"

# Agar file nahi hai to header create karo
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time", "Status"])

# Already marked students
marked_students = set()

camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera open nahi ho raha.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label_id, confidence = recognizer.predict(face)

        if confidence < 70:

            name = label_names.get(label_id, "Unknown")

            text = f"{name} - Recognized"

            # Attendance mark karo
            if name not in marked_students:

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        name,
                        date,
                        time,
                        "Present"
                    ])

                marked_students.add(name)

                print(
                    f"Attendance marked: {name} | {date} | {time}"
                )

        else:

            name = "Unknown"
            text = "Unknown"

        # Face box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "AI Attendance System",
        frame
    )

    # Q press = close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Attendance system closed.")