# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 2: Descripcion de puntos de interes mediante histogramas.

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea2

# Incluya aqui las librerias que necesite en su codigo
# ...
import matplotlib.pyplot as plt
import scipy.ndimage as nd

def descripcion_puntos_interes(imagen, coords_esquinas, vtam = 8, nbins = 16, tipoDesc='hist'):
    """
    # Esta funcion describe puntos de interes de una imagen mediante histogramas, analizando 
    # vecindarios con dimensiones "vtam+1"x"vtam+1" centrados en las coordenadas de cada punto de interes
    #   
    # La descripcion obtenida depende del parametro 'tipoDesc'
    #   - Caso 'hist': histograma normalizado de valores de gris 
    #   - Caso 'mag-ori': histograma de orientaciones de gradiente
    #
    # En el caso de que existan puntos de interes en los bordes de la imagen, el descriptor no
    # se calcula y el punto de interes se elimina de la lista <new_coords_esquinas> que devuelve
    # esta funcion. Esta lista indica los puntos de interes para los cuales existe descriptor.
    #
    # Argumentos de entrada:
    #   - imagen: numpy array con dimensiones [imagen_height, imagen_width].        
    #   - coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2] con las coordenadas 
    #                      de los puntos de interes detectados en la imagen. Tipo int64
    #                      Cada punto de interes se encuentra en el formato [fila, columna]
    #   - vtam: valor de tipo entero que indica el tamaño del vecindario a considerar para
    #           calcular el descriptor correspondiente.
    #   - nbins: valor de tipo entero que indica el numero de niveles que tiene el histograma 
    #           para calcular el descriptor correspondiente.
    #   - tipoDesc: cadena de caracteres que indica el tipo de descriptor calculado
    #
    # Argumentos de salida
    #   - descriptores: numpy array con dimensiones [num_puntos_interes, nbins] con los descriptores 
    #                   de cada punto de interes (i.e. histograma de niveles de gris)
    #   - new_coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2], solamente con las coordenadas 
    #                      de los puntos de interes descritos. Tipo int64  <class 'numpy.ndarray'>
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada vtam y nbins, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """

    # Inicializar la lista de descriptores
    descriptores = []

    # Verificar que el tamaño de la ventana sea par, si no lo es, terminar la función
    if vtam % 2 != 0:
        return

    # Convertir la imagen a formato flotante en el rango [0,1] si no está en ese rango.
    imagen = imagen.astype("float") / 255 if imagen.max() > 1 else imagen.astype("float")

    # Dimensiones de la imagen
    nrows, ncols = imagen.shape

    # Calcular el radio de la ventana de vecinos
    radio = int((vtam + 1) / 2)

    # Filtrar las coordenadas para excluir las esquinas cercanas a los bordes de la imagen.
    new_coords_esquinas = [
        (tempx, tempy) for tempx, tempy in coords_esquinas
        if (tempx + radio < nrows and tempx - radio >= 0 and
            tempy + radio < ncols and tempy - radio >= 0)
    ]

    # Calcular los descriptores en función del tipo solicitado
    if tipoDesc == "hist":
        # Crear los intervalos de los bins para el histograma
        bins = np.linspace(0, 1, nbins + 1)

        for tempx, tempy in new_coords_esquinas:
            # Definir la subimagen del vecindario
            x_start, y_start = tempx - radio, tempy - radio
            vecindario = imagen[x_start:x_start + vtam + 1, y_start:y_start + vtam + 1]

            # Calcular y normalizar el histograma
            hist, _ = np.histogram(vecindario.flatten(), bins=bins)
            hist = hist / hist.sum()
            descriptores.append(hist)

    elif tipoDesc == "mag-ori":
        # Calcular las derivadas de la imagen
        img_dx = nd.sobel(imagen, axis=0)
        img_dy = nd.sobel(imagen, axis=1)

        # Calcular la orientación y magnitud
        img_ori = (np.rad2deg(np.arctan2(img_dx, img_dy)) + 360) % 360
        img_mag = np.sqrt(img_dx ** 2 + img_dy ** 2)

        # Crear los intervalos de los bins para las orientaciones
        bins = np.linspace(0, 360, num=(nbins + 1))

        for tempx, tempy in new_coords_esquinas:
            # Definir la subimagen del vecindario
            x_start, y_start = tempx - radio, tempy - radio
            vec_ori = img_ori[x_start:x_start + vtam + 1, y_start:y_start + vtam + 1]
            vec_mag = img_mag[x_start:x_start + vtam + 1, y_start:y_start + vtam + 1]

            # Calcular el histograma ponderado por magnitud
            histograma = np.zeros(shape=nbins)
            indices = np.digitize(vec_ori, bins) - 1

            for mag, indx_bin in zip(vec_mag.flatten(), indices.flatten()):
                histograma[indx_bin] += mag

            descriptores.append(histograma)

    # Devolver los descriptores y las nuevas coordenadas de las esquinas
    return np.asarray(descriptores), np.asarray(new_coords_esquinas)


if __name__ == "__main__":    
    print("Practica 2 - Tarea 2 - Test autoevaluación\n")                

    ## tests descriptor tipo 'hist' (tarea 2a)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test
    #print("Tests completados = " + str(test_p2_tarea2(disptime=1,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist',imgIdx = 3, poiIdx = 7))) #analizar solamente imagen #2 y esquina #7    

    ## tests descriptor tipo 'mag-ori' (tarea 2b)
    print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test
    #print("Tests completados = " + str(test_p2_tarea2(disptime=0.1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=1,stop_at_error=True,debug=True,tipoDesc='mag-ori',imgIdx = 3,poiIdx = 7))) #analizar solamente imagen #1 y esquina #7       