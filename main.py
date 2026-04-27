from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import streamlit as st
import tensorflow as tf

from src.config import load_config
from src.labels import is_healthy, load_class_names
from src.predict import predict_top_k


CONFIG = load_config()
HOME_IMAGE = Path("home_page.jpeg")


DISEASE_GUIDANCE = {
    "healthy": "No visible disease class was detected. Keep monitoring the plant and maintain good watering, airflow, and soil hygiene.",
    "disease": "Isolate affected leaves where possible, avoid overhead watering, and confirm the diagnosis with a local agricultural expert before applying treatment.",
}


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def model_available() -> bool:
    return CONFIG.model_path.exists()


def download_model_from_env() -> bool:
    model_url = os.getenv("PLANT_MODEL_URL")
    if not model_url:
        return False

    CONFIG.model_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(model_url, CONFIG.model_path)
    return CONFIG.model_path.exists()


@st.cache_data(show_spinner=False)
def get_class_names(class_names_path: str) -> list[str]:
    return load_class_names(class_names_path)


def render_home() -> None:
    st.title("Plant Disease Detection")
    if HOME_IMAGE.exists():
        st.image(str(HOME_IMAGE), use_container_width=True)

    st.markdown(
        """
        Upload a crop leaf image and get a disease prediction from a TensorFlow CNN trained
        on 38 PlantVillage-style crop health classes.

        This demo is designed for portfolio review: it shows the model output, confidence
        ranking, and practical caution around using ML predictions for agricultural decisions.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Classes", "38")
    col2.metric("Reported test accuracy", "95.23%")
    col3.metric("Model parameters", "15.05M")


def render_about() -> None:
    st.header("Project Overview")
    st.markdown(
        """
        This project classifies healthy and diseased crop leaves using a custom convolutional
        neural network. The training notebook uses an 80/20 split from the training directory
        for training and validation, then evaluates the final model on a separate test folder.

        **Local dataset layout**

        - `dataset-2/train`: 43,456 images across 38 classes
        - `dataset-2/test`: 10,849 images across 38 classes

        **Current reported results**

        - Validation accuracy: 95.16%
        - Best validation accuracy: 95.66%
        - Test accuracy: 95.23%
        """
    )

    st.info(
        "This model is best understood as a portfolio ML demo. Real-world farm images can differ from the training dataset, so predictions should be verified before treatment decisions."
    )


def render_prediction() -> None:
    st.header("Disease Prediction")

    if not model_available():
        try:
            with st.spinner("Downloading model artifact..."):
                downloaded = download_model_from_env()
        except (OSError, URLError) as error:
            st.error(f"Could not download the trained model: {error}")
            return

        if not downloaded:
            st.error(
                "The trained model file was not found. Train the model with `make train`, place `trained_model.keras` in the project root, or set `PLANT_MODEL_URL` to a downloadable model artifact."
            )
            return

    uploaded_file = st.file_uploader(
        "Choose a crop leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.warning("Upload an image to run prediction.")
        return

    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Running model inference..."):
            model = load_trained_model(str(CONFIG.model_path))
            predictions = predict_top_k(
                model=model,
                image_path=BytesIO(uploaded_file.getvalue()),
                image_size=CONFIG.image_size,
                class_names=get_class_names(str(CONFIG.class_names_path)),
                top_k=CONFIG.top_k,
            )

        top_prediction = predictions[0]
        top_label = str(top_prediction["class_name"])
        confidence = float(top_prediction["confidence"])

        st.success(f"Top prediction: {top_prediction['display_name']} ({confidence:.2%})")

        st.subheader("Confidence Ranking")
        for prediction in predictions:
            st.progress(float(prediction["confidence"]), text=f"{prediction['display_name']} - {float(prediction['confidence']):.2%}")

        st.subheader("Suggested Next Step")
        guidance_key = "healthy" if is_healthy(top_label) else "disease"
        st.write(DISEASE_GUIDANCE[guidance_key])


def main() -> None:
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Select page", ["Home", "About", "Disease Prediction"])

    if app_mode == "Home":
        render_home()
    elif app_mode == "About":
        render_about()
    else:
        render_prediction()


if __name__ == "__main__":
    main()
