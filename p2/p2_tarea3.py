# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 3:  Similitud y correspondencia de puntos de interes

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea3

# Incluya aqui las librerias que necesite en su codigo
# ...
import matplotlib.pyplot as plt
from skimage import io
from p2_tarea1 import detectar_puntos_interes_harris
from p2_tarea2 import descripcion_puntos_interes

def correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr='mindist',max_distancia=25):
    """
    # Esta funcion determina la correspondencias entre dos conjuntos de descriptores mediante
    # el calculo de la similitud entre los descriptores.
    #
    # El parametro 'tipoCorr' determina el criterio de similitud aplicado 
    # para establecer correspondencias entre pares de descriptores:
    #   - Criterio 'mindist': minima distancia euclidea entre descriptores 
    #                         menor que el umbral 'max_distancia'
    #  
    # Argumentos de entrada:
    #   - descriptores1: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 1.        
    #   - descriptores2: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 2.        
    #   - tipoCorr: cadena de caracteres que indica el tipo de criterio para establecer correspondencias
    #   - max_distancia: valor de tipo double o float utilizado por el criterio 'mindist' y 'nndr', 
    #                    que determina si se aceptan correspondencias entre descriptores 
    #                    con distancia minima menor que 'max_distancia' 
    #
    # Argumentos de salida
    #   - correspondencias: numpy array con dimensiones [numero_correspondencias, 2] de tipo int64 
    #                       que determina correspondencias entre descriptores de imagen 1 e imagen 2.
    #                       Por ejemplo: 
    #                       correspondencias[0,:]=[5,22] significa que el descriptor 5 de la imagen 1 
    #                                                  corresponde con el descriptor 22 de la imagen 2. 
    #                       correspondencias[1,:]=[6,23] significa que el descriptor 6 de la imagen 1 
    #                                                  corresponde con el descriptor 23 de la imagen 2.
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada tipoCorr y max_distancia, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    #
    # CONSIDERACIONES: 
    # 1) La funcion no debe permitir correspondencias de uno a varios descriptores. Es decir, 
    #   un descriptor de la imagen 1 no puede asignarse a multiples descriptores de la imagen 2 
    # 2) En el caso de que existan varios descriptores de la imagen 2 con la misma distancia minima 
    #    con algún descriptor de la imagen 1, seleccione el descriptor de la imagen 2 con 
    #    indice/posicion menor. Por ejemplo, si las correspondencias [5,22] y [5,23] tienen la misma
    #    distancia minima, seleccione [5,22] al ser el indice 22 menor que 23
    """
    correspondencias = np.empty(shape=[0, 2], dtype=np.int64)  # initialize the output variable with dtype int64
    matched_descriptors = set()  # to keep track of matched descriptors in imagen2

    for i, descriptor1 in enumerate(descriptores_imagen1):
        mejor_distancia = np.inf
        mejor_j = -1
        for j, descriptor2 in enumerate(descriptores_imagen2):
            if j in matched_descriptors:
                continue  # skip already matched descriptors
            distancia = np.linalg.norm(descriptor1 - descriptor2)
            if tipoCorr == "mindist":
                if distancia < mejor_distancia or (distancia == mejor_distancia and j < mejor_j):
                    mejor_distancia = distancia
                    mejor_j = j
            elif tipoCorr == "nndr":
                pass
            else:
                pass
        if mejor_distancia < max_distancia and mejor_j != -1:
            correspondencias = np.vstack([correspondencias, [i, mejor_j]])
            matched_descriptors.add(mejor_j)  # mark this descriptor as matched

    return correspondencias

if __name__ == "__main__":
    print("Practica 2 - Tarea 3 - Test autoevaluación\n")


    ## tests correspondencias tipo 'minDist' (tarea 3a)
    print("Tests completados = " + str(test_p2_tarea3(disptime=5, stop_at_error=False, debug=False, tipoDesc='mag-ori', tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist' y ver errores
    #print("Tests completados = " + str(test_p2_tarea3(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist',tipoCorr='mindist' ))) #analizar todas las imagenes con descriptor 'hist'
    #print("Tests completados = " + str(test_p2_tarea3(disptime=1,stop_at_error=False,debug=False,tipoDesc='mag-ori',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'mag-ori'
    
    
    #Tarea 3
    # Test hist 
    #print("Tests with histogram descriptor:")
    #test_p2_tarea3(disptime=5, stop_at_error=False, debug=False, tipoDesc='hist', tipoCorr='mindist')

    #Tests with histogram descriptor:
    #Tests with histogram descriptor:
    #Practica 2 - Tarea 3
    #Realizando tests para la funcion 'correspondencias_puntos_interes' de la tarea 3
    #La funcion es correcta si los resultados obtenidos tienen una tolerancia de 2 decimales con respecto a la salida correcta.

    # * Utilizando datos en fichero '.\test_data\p2_tarea3_hist_mindist.data'
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #0 test_basica transformada #0... detectadas 2/2 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #0 test_basica transformada #1... detectadas 4/4 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #0 test_basica transformada #2... detectadas 4/4 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #1 Tablero_8x8_10 transformada #0... detectadas 17/17 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #1 Tablero_8x8_10 transformada #1... detectadas 27/27 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #1 Tablero_8x8_10 transformada #2... detectadas 9/9 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #2 Astronaut transformada #0... detectadas 14/14 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #2 Astronaut transformada #1... detectadas 17/17 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #2 Astronaut transformada #2... detectadas 13/13 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #3 Cofee transformada #0... detectadas 6/6 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #3 Cofee transformada #1... detectadas 19/19 correspondencias
    #   Testeando correspondencias descriptores tipo HIST y distancia MINDIST para imagen #3 Cofee transformada #2... detectadas 7/7 correspondencias
    #
    # * Finalizado en 21.811 secs
    # * RESULTADO FINAL: 139/139 CORRESPONDENCIAS CORRECTAS ( 100.00% )
    #Tests completados = True

    #Process finished with exit code 0


    # Test mag-ori
    #print("\nTests with magnitude-orientation descriptor:")
    #test_p2_tarea3(disptime=5, stop_at_error=False, debug=False, tipoDesc='mag-ori', tipoCorr='mindist')

    #Tests with magnitude-orientation descriptor:
    #Practica 2 - Tarea 3
    #Realizando tests para la funcion 'correspondencias_puntos_interes' de la tarea 3
    #La funcion es correcta si los resultados obtenidos tienen una tolerancia de 2 decimales con respecto a la salida correcta.

    # * Utilizando datos en fichero '.\test_data\p2_tarea3_mag-ori_mindist.data'
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #0 test_basica transformada #0... detectadas 2/2 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #0 test_basica transformada #1... detectadas 4/4 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #0 test_basica transformada #2... detectadas 4/4 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #1 Tablero_8x8_10 transformada #0... detectadas 17/17 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #1 Tablero_8x8_10 transformada #1... detectadas 20/20 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #1 Tablero_8x8_10 transformada #2... detectadas 3/3 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #2 Astronaut transformada #0... detectadas 14/14 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #2 Astronaut transformada #1... detectadas 14/14 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #2 Astronaut transformada #2... detectadas 7/7 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #3 Cofee transformada #0... detectadas 6/6 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #3 Cofee transformada #1... detectadas 11/11 correspondencias
    #    Testeando correspondencias descriptores tipo MAG-ORI y distancia MINDIST para imagen #3 Cofee transformada #2... detectadas 4/4 correspondencias

    # * Finalizado en 22.142 secs
    # * RESULTADO FINAL: 106/106 CORRESPONDENCIAS CORRECTAS ( 100.00% )
    #Tests completados = True

    #Process finished with exit code 0
    
    
    #Tarea 2
    # Cargar las imágenes en escala de grises
    #imagen1 = io.imread("img/NotreDame1.jpg", as_gray=True)
    #imagen2 = io.imread("img/NotreDame2.jpg", as_gray=True)  # Cambia "otra_imagen.jpg" por el nombre de tu segunda imagen

    
    # Detectar y describir puntos de interés en ambas imágenes
    #coords_imagen1 = detectar_puntos_interes_harris(imagen1)
    #coords_imagen2 = detectar_puntos_interes_harris(imagen2)
    #descriptores_imagen1, coords_filtrados_imagen1 = descripcion_puntos_interes(imagen1, coords_imagen1, tipoDesc="hist")
    #descriptores_imagen2, coords_filtrados_imagen2 = descripcion_puntos_interes(imagen2, coords_imagen2, tipoDesc="hist")

    # Encontrar correspondencias entre los descriptores de las dos imágenes
    #correspondencias = correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2)

    # Crear una imagen combinada
    #altura = max(imagen1.shape[0], imagen2.shape[0])
    #ancho_total = imagen1.shape[1] + imagen2.shape[1]
    #imagen_combinada = np.zeros((altura, ancho_total))

    # Colocar las dos imágenes en la imagen combinada
    #imagen_combinada[:imagen1.shape[0], :imagen1.shape[1]] = imagen1
    #imagen_combinada[:imagen2.shape[0], imagen1.shape[1]:] = imagen2

    # Visualizar la imagen combinada con las correspondencias
    #plt.figure(figsize=(15, 8))
    #plt.imshow(imagen_combinada, cmap='gray')
    #plt.axis('off')

    # Desplazamiento en el eje x para la segunda imagen
    #desplazamiento_x = imagen1.shape[1]

    # Dibujar los puntos de interés en ambas imágenes
    #plt.scatter(coords_filtrados_imagen1[:, 1], coords_filtrados_imagen1[:, 0], c='red', s=10, label="Puntos de interés Imagen 1")
    #plt.scatter(coords_filtrados_imagen2[:, 1] + desplazamiento_x, coords_filtrados_imagen2[:, 0], c='blue', s=10, label="Puntos de interés Imagen 2")

    # Conectar puntos correspondientes
    #for (i, j) in correspondencias:
    #    coord_imagen1 = coords_filtrados_imagen1[i]
    #    coord_imagen2 = coords_filtrados_imagen2[j]

    #    # Dibujar línea entre puntos correspondientes
    #    plt.plot([coord_imagen1[1], coord_imagen2[1] + desplazamiento_x], [coord_imagen1[0], coord_imagen2[0]], color="yellow", linewidth=0.5)

    #plt.legend()
    #plt.tight_layout()
    #plt.show()