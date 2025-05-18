from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

from model import build_model  # Ensure this loads a model with only age_output

data_path = "ProcessedData"
image_files = [f for f in os.listdir(data_path) if f.endswith(".jpg")]

ages, images = [], []

for file in image_files:
    try:
        parts = file.split("_")
        if len(parts) != 2:
            continue
        age = int(parts[1].replace(".jpg", ""))

        img = load_img(os.path.join(data_path, file), target_size=(128, 128))
        img = img_to_array(img)
        images.append(img)
        ages.append(age)
    except Exception as e:
        print(f"Skipping {file}: {e}")
        continue

X = np.array(images) / 255.0
y_age = np.array(ages)

X_train, X_test, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

model = build_model()  # Make sure this model has only age output
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_age_train,
          validation_data=(X_test, y_age_test),
          batch_size=32,
          epochs=50,
          callbacks=[early_stop])

model.save('vgg16_age_model_ProcessedData.h5')

