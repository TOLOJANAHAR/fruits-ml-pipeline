from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import pandas as pd

try:
    model = joblib.load('models/fruit_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError as e:
    raise RuntimeError(
        "Les fichiers de modèle ou de scaler sont introuvables. "
        "Vérifiez le dossier 'models/'."
    ) from e

app = FastAPI(
    title="Fruit Clustering API",
    description="API permettant de classifier des fruits dans des clusters en fonction de leurs caractéristiques physiques.",
    version="1.0.0"
)


class FruitData(BaseModel):
    """Représente les caractéristiques physiques d'un seul fruit."""
    Feature1: float = Field(..., description="Première caractéristique du fruit (ex: Poids)", example=150.5)
    Feature2: float = Field(..., description="Deuxième caractéristique du fruit (ex: Taille/Diamètre)", example=7.2)


@app.get("/", tags=["Général"])
def home():
    """
    Page d'accueil de l'API.
    Permet de vérifier que le service est fonctionnel.
    """
    return {"message": "Fruit Clustering API active et opérationnelle"}


@app.post("/predict", tags=["Prédictions"])
def predict(data: FruitData):
    """
    Prédit le cluster pour **un seul** fruit.
    
    - **data**: Objet JSON contenant Feature1 et Feature2.
    - **Retourne**: Le numéro du cluster attribué.
    """
    features = np.array([[data.Feature1, data.Feature2]])
    
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    cluster_id = int(prediction[0])

    return {
        "cluster": cluster_id,
        "message": f"Ce fruit appartient au cluster numéro {cluster_id}"
    }


@app.post("/predict_batch", tags=["Prédictions"])
def predict_batch(data: List[FruitData]):
    """
    Prédit les clusters pour **une liste** de fruits envoyée en JSON.
    
    - **data**: Une liste d'objets contenant les caractéristiques.
    - **Retourne**: Un tableau contenant la liste des clusters dans le même ordre.
    """
    features = np.array([[item.Feature1, item.Feature2] for item in data])
    
    scaled = scaler.transform(features)
    predictions = model.predict(scaled)

    return {
        "clusters": [int(p) for p in predictions]
    }


@app.post("/predict_csv", tags=["Prédictions"])
async def predict_csv(file: UploadFile):
    """
    Prédit les clusters à partir d'un **fichier CSV** téléversé.
    
    - **file**: Fichier CSV, contenant 2 colonnes séparées par des virgules.
    - **Retourne**: Un tableau contenant la liste des clusters pour chaque ligne du fichier.
    """
    df = pd.read_csv(file.file, header=None, names=["Feature1", "Feature2"])

    features = df.values
    scaled = scaler.transform(features)
    predictions = model.predict(scaled)

    return {
        "clusters": [int(p) for p in predictions]
    }