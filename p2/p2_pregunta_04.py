# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta XX

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr='mindist', max_distancia=25, umbral_nndr=0.75):
    """
    Esta funcion determina las correspondencias entre dos conjuntos de descriptores mediante
    el calculo de la similitud entre los descriptores.
    
    El parametro 'tipoCorr' determina el criterio de similitud aplicado 
    para establecer correspondencias entre pares de descriptores:
    - Criterio 'mindist': minima distancia euclidea entre descriptores 
      menor que el umbral 'max_distancia'
    - Criterio 'nndr': Nearest Neighbor Distance Ratio (NNDR), comparando la relación entre 
      la distancia del descriptor más cercano y el segundo más cercano, usando un umbral umbral_nndr.
    
    Argumentos de entrada:
    - descriptores1: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
                     con los descriptores de los puntos de interes de la imagen 1.        
    - descriptores2: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
                     con los descriptores de los puntos de interes de la imagen 2.        
    - tipoCorr: cadena de caracteres que indica el tipo de criterio para establecer correspondencias
    - max_distancia: valor de tipo double o float utilizado por el criterio 'mindist' y 'nndr', 
                     que determina si se aceptan correspondencias entre descriptores 
                     con distancia minima menor que 'max_distancia' 
    - umbral_nndr: valor de tipo float utilizado en el criterio NNDR, que determina si la relación
                   entre las distancias más cercanas es aceptable.
    
    Argumentos de salida:
    - correspondencias: numpy array con dimensiones [numero_correspondencias, 2] de tipo int64 
                        que determina correspondencias entre descriptores de imagen 1 e imagen 2.
    """
    correspondencias = np.empty(shape=[0, 2], dtype=np.int64)  # Inicializa la variable de salida con tipo int64
    matched_descriptors = set()  # Para llevar un registro de los descriptores emparejados en imagen2

    for i, descriptor1 in enumerate(descriptores_imagen1):
        mejor_distancia = np.inf
        mejor_j = -1
        distancias_minimas = []  # Para almacenar distancias de todos los descriptores

        for j, descriptor2 in enumerate(descriptores_imagen2):
            if j in matched_descriptors:
                continue  # Salta los descriptores ya emparejados
            distancia = np.linalg.norm(descriptor1 - descriptor2)
            distancias_minimas.append((distancia, j))

        distancias_minimas.sort(key=lambda x: x[0])  # Ordena por distancia ascendente

        # Aplicar el criterio NNDR si es necesario
        if tipoCorr == "nndr":
            if len(distancias_minimas) >= 2:
                distancia_1, j1 = distancias_minimas[0]
                distancia_2, j2 = distancias_minimas[1]
                ratio = distancia_1 / distancia_2

                # Comprobar si el ratio es menor que el umbral
                if ratio < umbral_nndr:
                    if distancia_1 < max_distancia and j1 not in matched_descriptors:
                        correspondencias = np.vstack([correspondencias, [i, j1]])
                        matched_descriptors.add(j1)  # Marca este descriptor como emparejado
        else:
            # Aplicar el criterio 'mindist'
            if distancias_minimas:
                distancia_minima, mejor_j = distancias_minimas[0]
                if distancia_minima < max_distancia and mejor_j not in matched_descriptors:
                    correspondencias = np.vstack([correspondencias, [i, mejor_j]])
                    matched_descriptors.add(mejor_j)  # Marca este descriptor como emparejado

    return correspondencias


# Función para visualizar las correspondencias
def mostrar_correspondencias(imagen1, imagen2, correspondencias, coords_imagen1, coords_imagen2):
    """
    Muestra las correspondencias entre los puntos de interés de dos imágenes usando matplotlib.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(imagen1, cmap='gray')
    ax[0].scatter(coords_imagen1[:, 1], coords_imagen1[:, 0], c='r', label='Puntos Imagen 1')
    ax[0].set_title("Imagen 1")

    ax[1].imshow(imagen2, cmap='gray')
    ax[1].scatter(coords_imagen2[:, 1], coords_imagen2[:, 0], c='g', label='Puntos Imagen 2')
    ax[1].set_title("Imagen 2")

    for corr in correspondencias:
        y1, x1 = coords_imagen1[corr[0]]
        y2, x2 = coords_imagen2[corr[1]]
        ax[0].plot([x1, x2], [y1, y2], 'b-')
        ax[1].plot([x1, x2], [y1, y2], 'b-')

    plt.show()


# Bloque principal
if __name__ == "__main__":
    # Cargar las imágenes de ejemplo
    imagen1 = data.camera()  # Imagen 1
    imagen2 = data.camera()  # Imagen 2 (puedes usar otra imagen diferente para la comparación)
    
    # Coordenadas de ejemplo de los puntos de interés (estas deben ser obtenidas con un detector de puntos de interés real)
    coords_imagen1 = np.array([[50, 50], [100, 100], [150, 150]])  # Ejemplo de coordenadas de puntos de interés
    coords_imagen2 = np.array([[55, 55], [95, 95], [145, 145]])  # Ejemplo de coordenadas de puntos de interés

    # Descriptores de ejemplo (estos deben ser calculados utilizando algún algoritmo como SIFT, ORB, etc.)
    descriptores_imagen1 = np.random.rand(len(coords_imagen1), 128)  # Simulación de descriptores
    descriptores_imagen2 = np.random.rand(len(coords_imagen2), 128)  # Simulación de descriptores

    # Calcular las correspondencias utilizando NNDR
    correspondencias = correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr="nndr", max_distancia=25, umbral_nndr=0.75)

    # Mostrar las correspondencias
    mostrar_correspondencias(imagen1, imagen2, correspondencias, coords_imagen1, coords_imagen2)
