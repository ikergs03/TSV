# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 04

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from p2_tarea1 import detectar_puntos_interes_harris
from p2_tarea2 import descripcion_puntos_interes

def correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr='mindist', max_distancia=25, nndr_threshold=0.75):
    """
    Esta función determina las correspondencias entre dos conjuntos de descriptores mediante
    el cálculo de la similitud entre los descriptores.

    Se utiliza el criterio 'mindist' para mínima distancia euclídea y 'nndr' para el Nearest Neighbor Distance Ratio.
    
    Argumentos:
        - descriptores_imagen1: descriptores de los puntos de interés de la imagen 1.
        - descriptores_imagen2: descriptores de los puntos de interés de la imagen 2.
        - tipoCorr: tipo de criterio para establecer correspondencias ('mindist' o 'nndr').
        - max_distancia: valor de umbral de distancia para 'mindist'.
        - nndr_threshold: umbral de NNDR para filtrar correspondencias en el caso de 'nndr'.
        
    Retorna:
        - correspondencias: correspondencias entre puntos de imagen 1 e imagen 2.
    """
    correspondencias = np.empty(shape=[0, 2], dtype=np.int64)  # Inicializar
    matched_descriptors = set()  # Para evitar correspondencias repetidas

    for i, descriptor1 in enumerate(descriptores_imagen1):
        # Inicializar las mejores distancias y las posiciones de los descriptores más cercanos
        distancias = []
        for j, descriptor2 in enumerate(descriptores_imagen2):
            distancia = np.linalg.norm(descriptor1 - descriptor2)  # Distancia euclídea
            distancias.append((j, distancia))

        # Ordenar por distancia
        distancias.sort(key=lambda x: x[1])

        if tipoCorr == "mindist":
            mejor_j, mejor_distancia = distancias[0]
            if mejor_distancia < max_distancia and mejor_j not in matched_descriptors:
                correspondencias = np.vstack([correspondencias, [i, mejor_j]])
                matched_descriptors.add(mejor_j)

        elif tipoCorr == "nndr":
            # Comprobar el ratio NNDR: distancia del vecino más cercano / distancia del segundo vecino más cercano
            mejor_j, mejor_distancia = distancias[0]
            segundo_j, segundo_mejor_distancia = distancias[1]
            nndr_ratio = mejor_distancia / segundo_mejor_distancia

            if nndr_ratio < nndr_threshold and mejor_j not in matched_descriptors:
                correspondencias = np.vstack([correspondencias, [i, mejor_j]])
                matched_descriptors.add(mejor_j)

    return correspondencias


# Bloque principal
if __name__ == "__main__":
    # Visualizar las correspondencias
    imagen1 = io.imread("img/NotreDame1.jpg", as_gray=True)
    imagen2 = io.imread("img/NotreDame2.jpg", as_gray=True)  # Cambiar por el nombre de tu imagen 2

    # Detectar los puntos de interés y describirlos
    coords_imagen1 = detectar_puntos_interes_harris(imagen1)
    coords_imagen2 = detectar_puntos_interes_harris(imagen2)
    descriptores_imagen1, coords_filtrados_imagen1 = descripcion_puntos_interes(imagen1, coords_imagen1, tipoDesc="hist")
    descriptores_imagen2, coords_filtrados_imagen2 = descripcion_puntos_interes(imagen2, coords_imagen2, tipoDesc="hist")

    # Encontrar las correspondencias usando NNDR
    correspondencias = correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr="nndr", nndr_threshold=0.75)

    # Crear una imagen combinada
    altura = max(imagen1.shape[0], imagen2.shape[0])
    ancho_total = imagen1.shape[1] + imagen2.shape[1]
    imagen_combinada = np.zeros((altura, ancho_total))

    # Colocar las imágenes
    imagen_combinada[:imagen1.shape[0], :imagen1.shape[1]] = imagen1
    imagen_combinada[:imagen2.shape[0], imagen1.shape[1]:] = imagen2

    # Mostrar la imagen combinada con las correspondencias
    plt.figure(figsize=(15, 8))
    plt.imshow(imagen_combinada, cmap='gray')
    plt.axis('off')

    # Desplazamiento en el eje x para la segunda imagen
    desplazamiento_x = imagen1.shape[1]

    # Dibujar los puntos de interés
    plt.scatter(coords_filtrados_imagen1[:, 1], coords_filtrados_imagen1[:, 0], c='red', s=10, label="Imagen 1")
    plt.scatter(coords_filtrados_imagen2[:, 1] + desplazamiento_x, coords_filtrados_imagen2[:, 0], c='blue', s=10, label="Imagen 2")

    # Conectar puntos correspondientes
    for (i, j) in correspondencias:
        coord_imagen1 = coords_filtrados_imagen1[i]
        coord_imagen2 = coords_filtrados_imagen2[j]
        plt.plot([coord_imagen1[1], coord_imagen2[1] + desplazamiento_x], [coord_imagen1[0], coord_imagen2[0]], color="yellow", linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()
