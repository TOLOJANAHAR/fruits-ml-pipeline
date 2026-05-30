import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)
from sklearn.metrics import silhouette_score

# Charger dataset
df = pd.read_csv(
    'data/fruits.csv',
    header=None,
    names=['Feature1', 'Feature2']
)

# Features
X = df

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèles
models = {
    'KMeans': KMeans(n_clusters=3, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5)
}

results = []

mlflow.set_experiment('fruit-clustering')

best_score = -1
best_model = None

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        clusters = model.fit_predict(X_scaled)

        unique_clusters = len(set(clusters))

        if unique_clusters > 1:

            score = silhouette_score(X_scaled, clusters)

            mlflow.log_param('model_name', name)
            mlflow.log_metric('silhouette_score', score)

            if name == 'KMeans':
                mlflow.log_metric('inertia', model.inertia_)

            results.append({
                'model': name,
                'score': score
            })

            print(f'{name} score : {score}')

            if score > best_score:
                best_score = score
                best_model = model

# Sauvegarde
joblib.dump(best_model, 'models/fruit_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print('Meilleur modèle sauvegardé !')

