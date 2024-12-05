# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW/BOF
# Tarea 2: extraccion de caracteristicas

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

from p3_tests import test_p3_tarea2

import numpy as np
from skimage import io, color, transform, feature
from skimage.feature import hog


def obtener_features_tiny(path_imagenes, tamano=16):
    list_img_desc_tiny = []

    for path in path_imagenes:
        img = io.imread(path)
        img_gray = color.rgb2gray(img) if len(img.shape) == 3 else img
        img_resized = transform.resize(img_gray, (tamano, tamano), anti_aliasing=True)
        img_flat = img_resized.flatten().reshape(1, -1)
        list_img_desc_tiny.append(img_flat)

    return list_img_desc_tiny

def obtener_features_hog(path_imagenes, tamano=100, orientaciones=9, pixeles_por_celda=(8, 8), celdas_bloque=(2, 2)):
    list_img_desc_hog = []

    for path in path_imagenes:
        # Leer imagen
        img = io.imread(path)

        # Convertir a escala de grises y normalizar a rango [0, 1]
        img_gray = color.rgb2gray(img) if len(img.shape) == 3 else img
        img_gray = img_gray / 255.0 if img_gray.max() > 1 else img_gray

        # Redimensionar al tamaño especificado
        img_resized = transform.resize(img_gray, (tamano, tamano), anti_aliasing=True)

        # Calcular características HOG
        features_hog = feature.hog(
            img_resized,
            orientations=orientaciones,
            pixels_per_cell=pixeles_por_celda,
            cells_per_block=celdas_bloque,
            feature_vector=False  # Mantener la estructura multidimensional
        )

        # Verificar si HOG produjo resultados
        if features_hog.size == 0:
            print(f"No se generaron características HOG para la imagen {path}")
            continue

        # Aplanar celdas y bloques en un vector (M, D)
        n_bloques_y, n_bloques_x, celdas_bloque_y, celdas_bloque_x, orientaciones = features_hog.shape
        features_hog_flat = features_hog.reshape(
            (n_bloques_y * n_bloques_x, celdas_bloque_y * celdas_bloque_x * orientaciones))

        # Agregar a la lista
        list_img_desc_hog.append(features_hog_flat)
    return list_img_desc_hog

if __name__ == "__main__":
    dataset_path = 'C:\\Users\\2alex\\PycharmProjects\\TSV\\p3\\dataset_scenes15\\scenes15'
    print("Practica 3 - Tarea 2 - Test autoevaluación\n")
    #print("Tests completados = " + str(test_p3_tarea2(dataset_path, stop_at_error=False, debug=False)))  # analizar todos los casos sin pararse en errores ni mostrar datos
    print("Tests completados = " + str(test_p3_tarea2(dataset_path, stop_at_error=True, debug=True)))  # analizar todos los casos, pararse en errores y mostrar datos
