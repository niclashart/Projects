from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get('/ping')
async def ping():
    return 'Hello, I am alive'

def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/ping')
async def predict(file: UploadFile = File(...)):
    image = read_files_as_image(await file.read())
    
    img_batch = np.expand_dims(image, 0)
    
    prediction = MODEL.predict(img_batch)
    pass

if __name__ =="__main__":
    uvicorn.run(app, host='localhost', port=8000)