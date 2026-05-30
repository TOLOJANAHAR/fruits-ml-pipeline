from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/fruit_model.pkl')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI()

class FruitData(BaseModel):
    Feature1: float
    Feature2: float

@app.get("/")
def home():
    return {"message": "Fruit Clustering API"}

@app.post("/predict")
def predict(data: FruitData):

    features = np.array([[data.Feature1, data.Feature2]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    return {
        "cluster": int(prediction[0]),
        "message": f"Ce point appartient au groupe {prediction[0]}"
    }

@app.post("/predict_batch")
def predict_batch(data: List[FruitData]):

    features = np.array([[item.Feature1, item.Feature2] for item in data])
    scaled = scaler.transform(features)
    predictions = model.predict(scaled)

    return {
        "clusters": [int(p) for p in predictions]
    }

@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):

    # lire csv
    df = pd.read_csv(file.file, header=None, names=["Feature1", "Feature2"])

    # transformer
    features = df.values
    scaled = scaler.transform(features)

    # prediction
    predictions = model.predict(scaled)

    return {
        "clusters": [int(p) for p in predictions]
    }