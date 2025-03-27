
El **Spectral Clustering** (Clustering Espectral) es un algoritmo basado en teoría de grafos que usa la descomposición en valores propios (eigenvalues) de una matriz de afinidad para realizar la agrupación de datos. Se diferencia de otros métodos como K-Means porque puede detectar clusters no convexos y estructuras más complejas en los datos.  

### **Pasos del Algoritmo**  
![image](https://github.com/user-attachments/assets/d4e68e07-5d3b-43ab-941e-f36b1eac46e4)


---

## **Implementación en C++ y PyTorch con Pybind11**  

Este programa implementa el algoritmo en C++ utilizando **PyTorch (libtorch) y Pybind11**, delegando todo el cálculo pesado a C++ y generando gráficas en Python.  

### **Implementación en C++ (`spectral_clustering.cpp`)**  
1. **Cálculo de la matriz de afinidad \( A \)** usando el kernel Gaussiano.  
2. **Cálculo de la matriz Laplaciana \( L = D - A \)**.  
3. **Descomposición en valores propios** usando `torch::linalg::eigh()`.  
4. **Reducción de dimensionalidad** seleccionando los \( k \) primeros eigenvectores.  
5. **Aplicación de K-Means** en los vectores propios para asignar etiquetas.  

El código en C++ está vinculado a Python usando **Pybind11**, lo que permite llamar la función `spectral_clustering()` desde Python.  

### **Implementación en Python (`main.py`)**  
1. Genera un conjunto de datos de prueba con `make_blobs()`.  
2. Convierte los datos a tensores de PyTorch y llama a `spectral_clustering()` en C++.  
3. Genera **dos gráficos**:  
   - **Dispersión de los clusters obtenidos.**  
   - **Gráfico de pérdida** (distancia de cada punto al centroide de su cluster).  

---

## **Ventajas de esta Implementación**  
✅ **Rendimiento optimizado**: Todo el cálculo de valores propios y clustering se hace en C++.  
✅ **Flexibilidad**: Puede usarse con diferentes conjuntos de datos sin cambiar la estructura del código.  
✅ **Uso de PyTorch**: Se integra fácilmente con otros modelos de Machine Learning.  
