from flask import Flask, request, render_template
import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('brain_tumor.h5')
class_names = ['Kein Tumor', 'Tumor']

# Erstelle das Verzeichnis 'uploads', falls es nicht existiert
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Keine Datei hochgeladen', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'Keine Datei ausgewählt', 400
        
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            result = predict_tumor(file_path)
            
            return render_template('result.html', prediction=result)
    
    return render_template('upload.html')

# Funktion zur Bildvorhersage
def predict_tumor(image_path):
    # Öffne das Bild und skaliere es auf die richtige Größe (224x224)
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Passe die Größe an die erwartete Eingabegröße des Modells an
    
    # Wandle das Bild in ein Array um und normalisiere die Werte
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Erstelle eine Batch-Dimension
    
    # Mache eine Vorhersage
    prediction = model.predict(image_array)
    
    # Konvertiere die Vorhersage in die Klasse (0 oder 1)
    predicted_class = class_names[int(np.round(prediction[0]))]
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)