import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from utils import preprocess_image_for_mnist

st.set_page_config(page_title="Digit Recognition", page_icon="🧠")

st.title("🧠 Handwritten Digit Recognition")
st.write("Upload a picture of a handwritten digit (0–9) and the AI will predict it.")

# Load CNN model
@st.cache_resource
def load_cnn_model():
    model = load_model("cnn_model.h5")
    return model

model = load_cnn_model()

uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Uploaded Image")
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Input Image", width=200)

with col2:
    st.subheader("Prediction Result")
    if uploaded:
        pil_img = Image.open(uploaded)
        processed = preprocess_image_for_mnist(pil_img)

        # Predict
        probs = model.predict(processed)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

        st.success(f"Predicted Digit: **{pred}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")

        # Show probability bar chart
        st.bar_chart(probs)

    else:
        st.info("Upload an image to see prediction.")

st.markdown("---")
st.caption("Model: CNN trained on MNIST | Author: Anh Vainionpää")
