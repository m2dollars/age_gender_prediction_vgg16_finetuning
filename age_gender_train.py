from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from model import build_model
import os
import pickle

# Load metadata
data_path = "UTKFace"
image_files = [f for f in os.listdir(data_path) if f.endswith(".jpg")]

ages, genders, images = [], [], []

for file in image_files:
    age, gender = file.split("_")[:2]
    try:
        img = load_img(os.path.join(data_path, file), target_size=(128, 128))  # RGB
        img = img_to_array(img)
        images.append(img)
        ages.append(float(age))
        genders.append(int(gender))
    except Exception as e:
        continue

X = np.array(images, dtype=np.float32) / 255.0
y_age = np.array(ages, dtype=np.float32)
y_gender = np.array(genders, dtype=np.int32)

X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    X, y_age, y_gender, test_size=0.2, random_state=42)

# Build model
model = build_model()

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, {'gender_output': y_gender_train, 'age_output': y_age_train},
                    validation_data=(X_test, {'gender_output': y_gender_test, 'age_output': y_age_test}),
                    batch_size=32, epochs=20, callbacks=[early_stop])

# Save the trained model
model.save('vgg16_age_gender_model_UTKFace.h5')

# Save training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

