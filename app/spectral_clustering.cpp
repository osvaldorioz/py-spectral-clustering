#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// Función para calcular la matriz de similitud (kernel Gaussiano)
torch::Tensor compute_similarity_matrix(const torch::Tensor& data, double sigma=1.0) {
    int n = data.size(0);
    torch::Tensor sim_matrix = torch::zeros({n, n});

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double dist = torch::norm(data[i] - data[j]).item<double>();
                sim_matrix[i][j] = exp(-dist * dist / (2 * sigma * sigma));
            }
        }
    }
    return sim_matrix;
}

// Función para calcular la matriz Laplaciana
torch::Tensor compute_laplacian(const torch::Tensor& similarity_matrix) {
    torch::Tensor D = similarity_matrix.sum(1).diag();  // Diagonal de la matriz de grados
    return D - similarity_matrix;  // Matriz Laplaciana
}

// Función para realizar el Spectral Clustering utilizando SVD en lugar de eigh
torch::Tensor spectral_clustering(const torch::Tensor& data, int num_clusters) {
    // Paso 1: Calcular la matriz de similitud
    torch::Tensor similarity_matrix = compute_similarity_matrix(data);

    // Paso 2: Calcular la matriz Laplaciana
    torch::Tensor laplacian_matrix = compute_laplacian(similarity_matrix);

    // Paso 3: Calcular los valores y vectores propios mediante SVD
    torch::Tensor U, S, V;
    std::tie(U, S, V) = torch::svd(laplacian_matrix);

    // Paso 4: Seleccionar los primeros `num_clusters` vectores propios
    torch::Tensor selected_eigenvectors = U.slice(1, 0, num_clusters);

    // Paso 5: Normalizar los vectores propios
    selected_eigenvectors = selected_eigenvectors / selected_eigenvectors.norm(2, 1).unsqueeze(1);

    // Paso 6: Aplicar K-means sobre los vectores propios seleccionados
    torch::Tensor centroids = selected_eigenvectors.slice(0, 0, num_clusters);
    return centroids;  // Para propósitos del ejemplo, retornamos los centroides
}

PYBIND11_MODULE(spectral_clustering, m) {
    m.def("spectral_clustering", &spectral_clustering, "Spectral Clustering using PyTorch");
}
