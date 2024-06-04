from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("")

@app.get('/ping')
async def ping():
    return 'Hello, I am alive'

def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/ping')
async def predict(file: UploadFile = File(...)):
    bytes = file.read()

if __name__ =="__main__":
    uvicorn.run(app, host='localhost', port=8000)