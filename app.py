import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import predict_age_gender
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Age & Gender Prediction", layout="wide")
st.title("Age and Gender Prediction")

uploaded_file = st.file_uploader("Upload a portrait image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    result_img, predictions = predict_age_gender(bgr_img)

    # Use Streamlit columns to show original and predicted side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Predicted Image")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    if predictions:
        # st.success(f"Predicted Gender: {predictions[0][0]} | Age: {predictions[0][1]}")
        st.success(f"Age : {predictions[0][1]}")
    else:
        st.warning("No face detected or multiple faces found.")

# Optional training history plot
if st.checkbox("Show training/validation plots"):
    try:
        with open("training_history.pkl", "rb") as f:
            history = pickle.load(f)

        # Gender classification accuracy
        st.subheader("Gender Classification Accuracy")
        plt.figure(figsize=(8, 4))
        plt.plot(history['gender_output_accuracy'], label='Train Accuracy')
        plt.plot(history['val_gender_output_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        # Age estimation MAE
        st.subheader("Age Estimation MAE")
        plt.figure(figsize=(8, 4))
        plt.plot(history['age_output_mae'], label='Train MAE')
        plt.plot(history['val_age_output_mae'], label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

    except FileNotFoundError:
        st.error("training_history.pkl not found.")
    except Exception as e:
        st.error(f"Error loading training history: {e}")
