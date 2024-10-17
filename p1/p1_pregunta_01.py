# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Memoria: codigo de la pregunta 01

# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# PAREJA/TURNO: 01/VIERNES


import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
from p1_tarea4 import run_fusion
from p1_utils import visualizar_lapl_piramide, visualizar_gaus_piramide

def cargar_imagenes(path_imagenes):
    # Rutas de las imágenes
    ruta_imgA = os.path.join(path_imagenes, "apple2.jpg")  
    ruta_imgB = os.path.join(path_imagenes, "orange2.jpg")  
    ruta_mask = os.path.join(path_imagenes, "mask_apple2_orange2.jpg")  

    # Cargar imágenes
    imgA = imageio.imread(ruta_imgA).astype(np.float64) / 255.0
    imgB = imageio.imread(ruta_imgB).astype(np.float64) / 255.0
    mask = imageio.imread(ruta_mask).astype(np.float64) / 255.0

    # Asegurarse de que todas las imágenes tengan el mismo tamaño
    if imgA.shape != imgB.shape or imgA.shape != mask.shape:
        raise ValueError("Las imágenes y la máscara deben tener el mismo tamaño.")

    return imgA, imgB, mask

def run_fusion_color(imgA, imgB, mask, niveles):
    # Descomponer en canales RGB
    canales_A = [imgA[:, :, i] for i in range(3)]
    canales_B = [imgB[:, :, i] for i in range(3)]
    canal_mask = mask[:, :, 0]  

    # Fusión de cada canal
    canales_fusionados = []
    for canalA, canalB in zip(canales_A, canales_B):
        _, _, _, _, _, _, canal_fusionado = run_fusion(canalA, canalB, canal_mask, niveles)
        canales_fusionados.append(canal_fusionado)

    # Juntar los canales fusionados para formar la imagen RGB
    img_fusionada = np.stack(canales_fusionados, axis=-1)

    # Asegurarse de que los valores estén en el rango [0, 1]
    img_fusionada = np.clip(img_fusionada, 0, 1)

    # Convertir de nuevo a rango [0, 255] para mostrar
    img_fusionada = (img_fusionada * 255).astype(np.uint8)

    return img_fusionada

def mostrar_imagen(imagen, titulo="Imagen"):
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    path_imagenes = "/home/e462135/TSV/p1/img/"
    
    # Cargar imágenes y máscara
    imgA, imgB, mask = cargar_imagenes(path_imagenes)
    
    # Número de niveles en la pirámide
    niveles = 1

    # Fusión de imágenes en color
    img_fusionada = run_fusion_color(imgA, imgB, mask, niveles)

    # Mostrar la imagen fusionada
    mostrar_imagen(img_fusionada, titulo="Imagen Fusionada")
