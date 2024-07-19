import tensorflow as tf
import numpy as np

class_names = ['PotatoEarly_blight', 'Potatohealthy', 'Potato___Late_blight']

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)

    predicted_label = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_label, confidence

#Example usage
img_path = 'Download.JPG'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))


model_path = "/home/niclas/Projects/DeepL_Potato_Disease/saved_models/Tomato.h5"
model = tf.keras.models.load_model(model_path)

predicted_label, confidence = predict(model, img)
print(f"Predicted class: {predicted_label}, Confidence: {confidence}%")

