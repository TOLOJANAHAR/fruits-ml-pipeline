# Fruit AI Clustering API

## Description

API de clustering basée sur un modèle de Machine Learning (KMeans et autres modèles testés) avec FastAPI.

---

## Application en ligne

API déployée sur Render :

- Base URL : https://fruits-ml-pipeline.onrender.com
- Swagger UI : https://fruits-ml-pipeline.onrender.com/docs

---

## Installation locale

### 1. Cloner le projet
```bash
git clone https://github.com/your-username/fruit-ai-pipeline.git
cd fruit-ai-pipeline
````

### 2. Lancer avec Docker

#### Build image

```bash
docker build -t fruit-ai .
```

#### Run container

```bash
docker run -p 8000:8000 fruit-ai
```

---

### 3. Test API

#### Endpoint

POST /predict

### Exemple requête

```json
{
  "Feature1": 43.7,
  "Feature2": 38.57
}
```

### Exemple réponse

```json
{
  "cluster": 0,
  "message": "Ce point appartient au groupe 0"
}
```