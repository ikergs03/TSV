# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW/BOF
# Tarea 1: modelo BOW/BOF

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

# librerias y paquetes por defecto
from p3_tests import test_p3_tarea1
import numpy as np

def construir_vocabulario(list_img_desc, vocab_size=5, max_iter=300):
    """   
    # Esta funcion utiliza K-Means para agrupar los descriptores en "vocab_size" clusters.
    #
    # Argumentos de entrada:
    # - list_array_desc: Lista 1xN con los descriptores de cada imagen. Cada posicion de la lista 
    #                   contiene (MxD) numpy arrays que representan UNO O VARIOS DESCRIPTORES 
    #                   extraidos sobre la imagen
    #                   - M es el numero de vectores de caracteristicas/features de cada imagen 
    #                   - D el numero de dimensiones del vector de caracteristicas/feature.    
    #   - vocab_size: int, numero de palabras para el vocabulario a construir.    
    #   - max_iter: int, numero maximo de iteraciones del algoritmo KMeans
    #
    # Argumentos de salida:
    #   - vocabulario: Numpy array de tamaño [vocab_size, D], 
    #                   que contiene los centros de los clusters obtenidos por K-Means
    #
    #
    # NOTA: se sugiere utilizar la funcion sklearn.cluster.KMeans
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html     
    """
    vocabulario = np.empty(shape=[vocab_size,list_img_desc[0].shape[1]]) # iniciamos la variable de salida (numpy array)

    #... 

    return vocabulario

def obtener_bags_of_words(list_img_desc, vocab):
    """
    # Esta funcion obtiene el Histograma Bag of Words para cada imagen
    #
    # Argumentos de entrada:
    # - list_img_desc: Lista 1xN con los descriptores de cada imagen. Cada posicion de la lista 
    #                   contiene (MxD) numpy arrays que representan UNO O VARIOS DESCRIPTORES 
    #                   extraidos sobre la imagen
    #                   - M es el numero de vectores de caracteristicas/features de cada imagen 
    #                   - D el numero de dimensiones del vector de caracteristicas/feature.  
    #   - vocab: Numpy array de tamaño [vocab_size, D], 
    #                  que contiene los centros de los clusters obtenidos por K-Means.   
    #
    # Argumentos de salida: 
    #   - list_img_bow: Array de Numpy [N x vocab_size], donde cada posicion contiene 
    #                   el histograma bag-of-words construido para cada imagen.
    #
    """
    # iniciamos la variable de salida (numpy array)
    list_img_bow = np.empty(shape=[len(list_img_desc),len(vocab)]) 
    
    #...
    
    return list_img_bow

if __name__ == "__main__":    
    dataset_path = './datasets/scenes15/'
    print("Practica 3 - Tarea 1 - Test autoevaluación\n")                    
    print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores ni mostrar datos
    #print("Tests completados = " + str(test_p3_tarea1(dataset_path,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar datos