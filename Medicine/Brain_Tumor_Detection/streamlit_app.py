import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Funktion zum Laden und Vorverarbeiten des Bildes
def load_and_preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Füge eine Batch-Dimension hinzu
    return img

# Lade dein Modell (ersetze den Pfad durch den tatsächlichen Pfad zu deinem Modell)
model = tf.keras.models.load_model('brain_tumor.h5')

# Streamlit App
st.title("Tumor Klassifikation")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lade das Bild
    image = Image.open(uploaded_file)
    
    # Zeige das Bild an
    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
    
    # Verarbeite das Bild
    processed_image = load_and_preprocess_image(image)
    
    # Mache eine Vorhersage
    prediction = model.predict(processed_image)
    result = 'Tumor' if prediction[0][0] > 0.5 else 'Kein Tumor'
    
    # Zeige das Ergebnis an
    st.write(f"Vorhersage: {result}")