import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("vgg16_age_gender_model_UTKFace.h5")

def predict_age_gender(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []

    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_arr = face_img.astype("float32") / 255.0
        face_arr = np.expand_dims(face_arr, axis=0)

        gender_pred, age_pred = model.predict(face_arr)
        gender = "Male" if np.argmax(gender_pred) == 1 else "Female"
        age = int(age_pred[0][0])

        # Draw
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{gender}, {age}"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        results.append((gender, age))

    return image, results
