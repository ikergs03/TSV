# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Tarea 4: fusion de imagenes

# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# PAREJA/TURNO: 01/VIERNES

import numpy as np
import math

from p1_tests import test_p1_tarea4
from p1_utils import visualizar_fusion
from p1_tarea1 import reduce, expand
from p1_tarea2 import gaus_piramide, lapl_piramide
from p1_tarea3 import fusionar_lapl_pyr, reconstruir_lapl_pyr



def run_fusion(imgA, imgB, mask, niveles): 
    """ 
    # Esta funcion implementa la fusion de dos imagenes calculando las 
    # pirámides Laplacianas de las imagenes de entrada y la pirámide
    # Gausiana de una mascara.
    #  
    # Argumentos de entrada:
    #   imgA: numpy array de tamaño [imagen_height, imagen_width].
    #   imgB: numpy array de tamaño [imagen_height, imagen_width].
    #   mask: numpy array de tamaño [imagen_height, imagen_width].
    #
    # Devuelve:
    #   Gpyr_imgA: lista de numpy arrays con variable tamaño con "niveles+1" elementos 
    #               correspodientes a la piramide Gaussiana de la imagen A
    #   Gpyr_imgB: lista de numpy arrays con variable tamaño con "niveles+1" elementos 
    #               correspodientes a la piramide Gaussiana de la imagen B
    #   Gpyr_mask: lista de numpy arrays con variable tamaño con "niveles+1" elementos 
    #               correspodientes a la piramide Gaussiana de la máscara
    #   Lpyr_imgA: lista de numpy arrays con variable tamaño con "niveles+1" elementos 
    #               correspodientes a la piramide Laplaciana de la imagen A
    #   Lpyr_imgB: lista de numpy arrays con variable tamaño con "niveles+1" elementos 
    #               correspodientes a la piramide Laplaciana de la imagen B
    #   Lpyr_fus: lista de numpy arrays con variable tamaño con "niveles+1" elementos 
    #               correspodientes a la piramide Laplaciana de la fusion imagen A & B
    #   Lpyr_fus_rec:  numpy array de tamaño [imagen_height, imagen_width] correspondiente
    #               a la reconstruccion de la pirámide Lpyr_fus
    """ 
    
    # iniciamos las variables de salida    
    Gpyr_imgA = []      # Pirámide Gaussiana imagen A
    Gpyr_imgB = []      # Pirámide Gaussiana imagen B
    Gpyr_mask = []      # Pirámide Gaussiana máscara    
    Lpyr_imgA = []      # Pirámide Laplaciana imagen A
    Lpyr_imgB = []      # Pirámide Laplaciana imagen B
    Lpyr_fus = []       # Pirámide Laplaciana fusionada
    Lpyr_fus_rec = []   # Imagen reconstruida de la pirámide Laplaciana fusionada

    #Verificar que las imágenes son matrices bidimensionales
    if len(imgA.shape) != 2 or len(imgB.shape) != 2 or len(mask.shape) != 2:
        raise ValueError("Las imágenes deben ser matrices bidimensionales.")


    # 1. Convertir las imágenes y la máscara a tipo float, normalizándolas en el rango [0,1]
    imgA = imgA.astype(np.float64)
    imgB = imgB.astype(np.float64)
    mask = mask.astype(np.float64)

    #Normalizar todas las imágenes/máscara tipo float en el rango [0,1]

    imgA = np.clip(imgA, 0, 1)
    imgB = np.clip(imgB, 0, 1)
    mask = np.clip(mask, 0, 1)


    # 2. Calcular las pirámides Gaussianas de las imágenes y la máscara
    Gpyr_imgA = gaus_piramide(imgA, niveles)
    Gpyr_imgB = gaus_piramide(imgB, niveles)
    Gpyr_mask = gaus_piramide(mask, niveles)

    # 3. Calcular las pirámides Laplacianas de las imágenes
    Lpyr_imgA = lapl_piramide(Gpyr_imgA)
    Lpyr_imgB = lapl_piramide(Gpyr_imgB)

    # 4. Fusionar las pirámides Laplacianas utilizando la pirámide Gaussiana de la máscara
    Lpyr_fus = fusionar_lapl_pyr(Lpyr_imgA, Lpyr_imgB, Gpyr_mask)

    # 5. Reconstruir la imagen fusionada a partir de la pirámide Laplaciana fusionada
    Lpyr_fus_rec = reconstruir_lapl_pyr(Lpyr_fus)

    # 6. Recortar los valores fuera del rango [0, 1]
    Lpyr_fus_rec = np.clip(Lpyr_fus_rec, 0, 1)
    
    return Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, Lpyr_fus_rec
if __name__ == "__main__":    
    
    path_imagenes = "./p1/img/"

    print("Practica 1 - Tarea 4 - Test autoevaluación\n")    
    result,imgAgray,imgBgray,maskgray,\
        Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, Lpyr_fus_rec \
            = test_p1_tarea4(path_img=path_imagenes,precision=2)
    print("Tests completado = " + str(result)) 

    if result==True:
        #visualizar piramides de la fusion (puede consultar el codigo en el fichero p1_utils.py)
        visualizar_fusion(imgAgray,imgBgray,maskgray,Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, Lpyr_fus_rec)