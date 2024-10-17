# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Memoria: codigo de la pregunta XX

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO


import numpy as np
import os

from p1_tests import test_p1_tarea4
from p1_utils import visualizar_fusion
from p1_tarea1 import reduce, expand
from p1_tarea2 import gaus_piramide, lapl_piramide
from p1_tarea3 import fusionar_lapl_pyr, reconstruir_lapl_pyr
from p1_tarea4 import run_fusion

def run_fusion_color(imgA, imgB, mask, niveles):
    # Asegúrate de que las imágenes y la máscara son del mismo tamaño
    if imgA.shape != imgB.shape or imgA.shape != mask.shape:
        raise ValueError("Las imágenes y la máscara deben tener el mismo tamaño.")

    # Inicializamos las variables de salida
    Gpyr_imgA = []
    Gpyr_imgB = []
    Gpyr_mask = []
    Lpyr_imgA = []
    Lpyr_imgB = []
    Lpyr_fus = []
    Lpyr_fus_rec = []

    # Convertir imágenes y máscara a tipo float y normalizarlas
    imgA = imgA.astype(np.float64) / 255.0
    imgB = imgB.astype(np.float64) / 255.0
    mask = mask.astype(np.float64) / 255.0

    # Calcular las pirámides Gaussianas
    Gpyr_imgA = gaus_piramide(imgA, niveles)
    Gpyr_imgB = gaus_piramide(imgB, niveles)
    Gpyr_mask = gaus_piramide(mask, niveles)

    # Calcular las pirámides Laplacianas
    Lpyr_imgA = lapl_piramide(Gpyr_imgA)
    Lpyr_imgB = lapl_piramide(Gpyr_imgB)

    # Fusionar las pirámides Laplacianas
    Lpyr_fus = fusionar_lapl_pyr(Lpyr_imgA, Lpyr_imgB, Gpyr_mask)

    # Reconstruir la imagen fusionada
    Lpyr_fus_rec = reconstruir_lapl_pyr(Lpyr_fus)

    # Recortar los valores fuera del rango [0, 1]
    Lpyr_fus_rec = np.clip(Lpyr_fus_rec, 0, 1)

    # Convertir de nuevo a rango [0, 255] para mostrar
    Lpyr_fus_rec = (Lpyr_fus_rec * 255).astype(np.uint8)

    return Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, Lpyr_fus_rec


if __name__ == "__main__":    
    
    path_imagenes = "/home/e462135/TSV/p1/img/"

    print("Practica 1 - Tarea 4 - Test autoevaluación\n")    
    result, imgAgray, imgBgray, maskgray, \
        Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, Lpyr_fus_rec \
            = test_p1_tarea4(path_img=path_imagenes, precision=2)
    print("Tests completado = " + str(result)) 

    if result == True:
        # Visualizar pirámides de la fusión
        visualizar_fusion(imgAgray, imgBgray, maskgray, Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, Lpyr_fus_rec)



#path_imagenes = "/home/e462135/TSV/p1/img/"