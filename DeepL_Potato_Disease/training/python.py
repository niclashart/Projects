import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from training import model


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 10

st.title('Plant Disease Classification using Deep Learning')

st.write('Anhand dieser App können Krankheiten von Tomaten-, Paprika- und Kartoffelpflanzen erkannt werden!')

upload_file = st.file_uploader('Wähle ein Bild aus', type=['png', 'jpg', 'jpeg'])

if upload_file is not None:
    
    image = Image.open(upload_file)
    
    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
    
    st.write('Bildformat: ', image.format)
    st.write('Bildgröße: ', image.size)
    st.write('Bildmodus: ', image.mode)

prediction = model.predict(image)
st.write(f'Vorhersage: {prediction}')