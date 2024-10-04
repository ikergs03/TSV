# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Tarea 1: metodos reduce y expand

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO
import numpy as np
import scipy.signal

from p1_tests import test_p1_tarea1
from p1_utils import generar_kernel_suavizado

def reduce(imagen):
    """  
    # Esta funcion implementa la operacion "reduce" sobre una imagen
    # 
    # Argumentos de entrada:
    #    imagen: numpy array de tamaño [imagen_height, imagen_width].
    # 
    # Devuelve:
    #    output: numpy array de tamaño [imagen_height/2, imagen_width/2] (output).
    #
    # NOTA: si imagen_height/2 o imagen_width/2 no son numeros enteros, 
    #        entonces se redondea al entero mas cercano por arriba 
    #        Por ejemplo, si la imagen es 5x7, la salida sera 3x4  
    """   
    output = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

    #...
   
    return output  

def expand(imagen):
    """  
    # Esta funcion implementa la operacion "expand" sobre una imagen
    # 
    # Argumentos de entrada:
    #    imagen: numpy array de tamaño [imagen_height, imagen_width].
    #     
    # Devuelve:
    #    output: numpy array de tamaño [imagen_height*2, imagen_width*2].
    """ 
    output = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

    #...

    return output

if __name__ == "__main__":    
    print("Practica 1 - Tarea 1 - Test autoevaluación\n")                
    print("Tests completados = " + str(test_p1_tarea1())) 