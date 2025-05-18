# age_gender_prediction_vgg16_finetuning

## Project Overview

This project implements an age and gender prediction system based on face images. It leverages transfer learning by fine-tuning a pre-trained VGG16 convolutional neural network (CNN) to simultaneously perform gender classification and age regression. The system is designed to:

- Detect faces in raw images,
- Crop and preprocess images,
- Predict age and gender from the face images,
- Provide a Streamlit web app UI for easy image upload and prediction visualization,
- Display training and validation metrics to monitor model performance.

---

## Features

- Face detection using OpenCV's Haar Cascades to preprocess input data.
- Fine-tuned VGG16 backbone for feature extraction.
- Dual-output model with:
  - Gender classification (binary classification),
  - Age estimation (regression).
- Early stopping during training to avoid overfitting.
- Streamlit-based UI for interactive predictions.
- Visualization of training history with accuracy and loss plots.
- Dataset processing pipeline that extracts faces and tags them with ID and computed age from raw filenames.

---

## Dataset

- Images are processed from a raw dataset folder where filenames are formatted as:
Example: `3999_1924-11-20_2010.jpg`

- The system extracts the age by subtracting DOB from the capture year.
- Images with zero or multiple detected faces are discarded.
- Cropped face images are resized to 128x128 pixels and saved with the format:

---

## Model Architecture

- **Backbone:** VGG16 pre-trained on ImageNet (without top layers).
- **Fine-tuning:** Last convolutional block layers are unfrozen to adapt to new task.
- **Output heads:**
- **Gender output:** Sigmoid activation for binary classification.
- **Age output:** Linear activation for continuous age prediction.
- **Losses:**
- Binary crossentropy for gender classification.
- Mean squared error (MSE) for age regression.

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Recommended: Create and activate a virtual environment
- NVIDIA GeForce RTX 4060 Ti

### Install dependencies

```bash
pip install -r requirements.txt

# Data Preparation
- Place your raw images in the RawData/ folder.
- Run the preprocessing script to detect faces, crop, and save processed images
----
python data_processing.py
----
# Model Training
python age_gender_train.py
python age_only_train.py
# Prediction & UI
streamlit run app.py

├── RawData/                  # Raw images folder (input)
├── ProcessedData/            # Cropped face images (processed dataset)
├── model.py                  # Model architecture definition
├── training.py               # Training script with early stopping
├── preprocess_faces.py       # Face detection and cropping script
├── predict.py                # Prediction logic script
├── app.py                   # Streamlit UI app
├── training_history.pkl      # Pickled training metrics
├── vgg16_age_gender_model.h5 # Trained model weights
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)



