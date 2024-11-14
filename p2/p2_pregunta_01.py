# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta XX

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, feature
from p2_tarea2 import descripcion_puntos_interes

# Cargar la imagen de la cámara
imagen = data.camera()

# Detectar las esquinas usando corner_harris y corner_peaks
esquinas = feature.corner_harris(imagen)
coords_esquinas = feature.corner_peaks(esquinas, min_distance=5)

# Tamaños de vecindario y números de bins a probar
vtams = [8, 16]
nbins_values = [16, 32]

# Crear una figura para visualizar los resultados
fig, axes = plt.subplots(len(vtams), len(nbins_values), figsize=(12, 12))

# Iterar sobre todas las combinaciones de vtam y nbins
for i, vtam in enumerate(vtams):
    for j, nbins in enumerate(nbins_values):
        # Obtener los descriptores y las nuevas coordenadas
        descriptores, new_coords_esquinas = descripcion_puntos_interes(imagen, coords_esquinas, vtam=vtam, nbins=nbins, tipoDesc='hist')
        
        # Mostrar el histograma de los primeros puntos de interés para cada combinación de parámetros
        ax = axes[i, j]
        ax.hist(descriptores[0], bins=nbins, range=(0, 1), density=True)
        ax.set_title(f'vtam={vtam}, nbins={nbins}')
        ax.set_xlabel('Valor del descriptor')
        ax.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()
