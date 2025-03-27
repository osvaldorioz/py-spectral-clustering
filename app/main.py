from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import spectral_clustering 

import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/spectral-clustering")
def calculo(n_samples: int, centers: int, random_state: int):
    output_file_1 = "dispersion.png"
    output_file_2 = "loss.png"
    # Definir correctamente los parámetros
    #n_samples = 300
    #centers = 3
    #random_state = 42

    # Generar datos sintéticos de ejemplo
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)

    # Convertir los datos a un tensor de PyTorch
    data = torch.tensor(X, dtype=torch.float32)

    # Realizar el clustering espectral
    num_clusters = centers
    centroids = spectral_clustering.spectral_clustering(data, num_clusters)

    # Mostrar las gráficas de dispersión
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title("Gráfico de dispersión de los datos")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.savefig(output_file_1)
    #plt.show()

    # Calcular los centroides como la media de los puntos de cada clúster
    centroids_calculados = []
    for i in range(num_clusters):
        cluster_points = data[y == i]  # Puntos del clúster
        centroid = cluster_points.mean(dim=0)  # Calcular el centroide como la media
        centroids_calculados.append(centroid)

    # Convertir la lista de centroides a un tensor
    centroids_calculados = torch.stack(centroids_calculados)

    # Mostrar las gráficas de pérdida (en este caso, la pérdida es la distancia entre los puntos y los centroides)
    distances = []
    for i in range(num_clusters):
        cluster_points = data[y == i]  # Puntos del clúster
        centroid = centroids_calculados[i]  # Centroide calculado del clúster
        dist = torch.norm(cluster_points - centroid, dim=1)  # Distancia al centroide
        distances.append(dist)

    # Unir todas las distancias
    distances = torch.cat(distances).numpy()

    # Graficar la pérdida
    plt.figure(figsize=(8, 6))
    plt.plot(distances, label="Pérdida")
    plt.title("Gráfico de pérdida")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Distancia al centroide")
    plt.legend()
    plt.savefig(output_file_2)
    #plt.show()
    plt.close()
    
    j1 = {
        "Dispersion plot": output_file_1, 
        "Loss plot": output_file_2
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/spectral-clustering-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)