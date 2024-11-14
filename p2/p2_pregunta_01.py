# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 01

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, feature
from scipy import ndimage as nd

# Función proporcionada para la descripción de puntos de interés
def descripcion_puntos_interes(imagen, coords_esquinas, vtam=8, nbins=16, tipoDesc='hist'):
    descriptores = []

    if vtam % 2 != 0:
        return

    imagen = imagen.astype("float") / 255 if imagen.max() > 1 else imagen.astype("float")
    nrows, ncols = imagen.shape
    radio = int((vtam + 1) / 2)

    new_coords_esquinas = [
        (tempx, tempy) for tempx, tempy in coords_esquinas
        if (tempx + radio < nrows and tempx - radio >= 0 and
            tempy + radio < ncols and tempy - radio >= 0)
    ]

    if tipoDesc == "hist":
        bins = np.linspace(0, 1, nbins + 1)
        for tempx, tempy in new_coords_esquinas:
            x_start, y_start = tempx - radio, tempy - radio
            vecindario = imagen[x_start:x_start + vtam + 1, y_start:y_start + vtam + 1]
            hist, _ = np.histogram(vecindario.flatten(), bins=bins)
            hist = hist / hist.sum()
            descriptores.append(hist)

    elif tipoDesc == "mag-ori":
        img_dx = nd.sobel(imagen, axis=0, mode='constant')
        img_dy = nd.sobel(imagen, axis=1, mode='constant')
        img_ori = (np.rad2deg(np.arctan2(img_dx, img_dy)) + 360) % 360
        img_mag = np.sqrt(img_dx ** 2 + img_dy ** 2)
        bins = np.linspace(0, 360, num=(nbins + 1))

        for tempx, tempy in new_coords_esquinas:
            x_start, y_start = tempx - radio, tempy - radio
            vec_ori = img_ori[x_start: x_start + vtam + 1, y_start: y_start + vtam + 1]
            vec_mag = img_mag[x_start: x_start + vtam + 1, y_start: y_start + vtam + 1]
            histograma, _ = np.histogram(vec_ori, bins=bins, weights=vec_mag)
            descriptores.append(histograma)

    return np.asarray(descriptores), np.asarray(new_coords_esquinas)

# Cargar la imagen y extraer puntos de interés
imagen = data.camera()
esquinas = feature.corner_harris(imagen)
coords_esquinas = feature.corner_peaks(esquinas, min_distance=3)

# Configuraciones para los histogramas
vtam_values = [8, 16]
nbins_values = [16, 32]

# Crear la figura para los histogramas
fig, axes = plt.subplots(len(vtam_values), len(nbins_values), figsize=(12, 8))
fig.suptitle("Histograma de descriptores para diferentes vtam y nbins", fontsize=16)

# Generar los histogramas para cada combinación de tamaño de vecindario y número de bins
for i, vtam in enumerate(vtam_values):
    for j, nbins in enumerate(nbins_values):
        # Obtener descriptores usando el tipo 'hist'
        descriptores, new_coords_esquinas = descripcion_puntos_interes(imagen, coords_esquinas, vtam=vtam, nbins=nbins, tipoDesc='hist')

        # Seleccionar el primer descriptor como muestra (o ajustar para ver más)
        ax = axes[i, j]
        ax.hist(descriptores[0], bins=nbins, range=(0, 1), density=True)
        ax.set_title(f'vtam={vtam}, nbins={nbins}')
        ax.set_xlabel("Valor del descriptor")
        ax.set_ylabel("Frecuencia")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
