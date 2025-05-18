import cv2
import os
import datetime
from pathlib import Path

input_folder = Path("RawData")
output_folder = Path("ProcessedData")
output_folder.mkdir(exist_ok=True)

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Process each Image
for file in input_folder.glob("*.jpg"):
    try:
        parts = file.stem.split("_")
        if len(parts) != 3:
            continue
        person_id = parts[0]
        DOB = datetime.datetime.strptime(parts[1], "%Y-%m-%d")
        capture_year = int(parts[2])
        
        age = capture_year - DOB.year
        if age < 0 or age > 120:
            continue
        # Read image
        img = cv2.imread(str(file))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) != 1:
            continue  # Skip images without exactly one face

        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))

        # Save with format ID_age.jpg
        output_filename = f"{person_id}_{age}.jpg"
        print(output_filename)
        cv2.imwrite(str(output_folder / output_filename), face_img)

    except Exception as e:
        print(f"Skipping {file.name}: {e}")