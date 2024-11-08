# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 1: Deteccion de puntos de interes con Harris corner detector.

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea1

# Incluya aqui las librerias que necesite en su codigo
# ...
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage.feature import corner_peaks


def detectar_puntos_interes_harris(imagen, sigma=1.0, k=0.05, threshold_rel=0.2):
    """
    Esta función detecta puntos de interés en una imagen con el algoritmo de Harris.

    Argumentos de entrada:
        - imagen: numpy array con dimensiones [imagen_height, imagen_width].
        - sigma: valor de tipo float para el suavizado aplicado.
        - k: valor de tipo float para calcular la respuesta R de Harris.
        - threshold_rel: valor de tipo float para el umbral relativo.

    Argumentos de salida:
        - coords_esquinas: numpy array [num_puntos_interes, 2] con las coordenadas de los puntos de interés
                           en el formato [fila, columna] de tipo int64.
    """
    # Normalización manual de la imagen en el rango [0,1]
    imagen = imagen.astype(np.float64)
    imagen /= imagen.max()

    # Derivadas de la imagen usando filtros Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    I_x = convolve2d(imagen, sobel_x, mode='same')
    I_y = convolve2d(imagen, sobel_y, mode='same')

    # Productos de derivadas en cada pixel
    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y

    # Suavizar los productos de derivadas con un filtro Gaussiano
    S_x2 = gaussian_filter(I_x2, sigma=sigma, mode='constant')
    S_y2 = gaussian_filter(I_y2, sigma=sigma, mode='constant')
    S_xy = gaussian_filter(I_xy, sigma=sigma, mode='constant')

    # Calcular la respuesta de Harris
    det_M = S_x2 * S_y2 - S_xy ** 2
    trace_M = S_x2 + S_y2
    R = det_M - k * (trace_M ** 2)

    # Aplicar umbral relativo
    R_max = R.max()
    threshold = threshold_rel * R_max
    R[R < threshold] = 0

    # Identificación de picos locales con un `min_distance` mayor
    coords_esquinas = corner_peaks(R, min_distance=5, threshold_rel=threshold_rel)

    return coords_esquinas

if __name__ == "__main__":    
    print("Practica 2 - Tarea 1 - Test autoevaluación\n")                
    
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=1,stop_at_error=False,debug=False))) #analizar y visualizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=False))) #analizar todos los casos y pararse en errores
    print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar informacion