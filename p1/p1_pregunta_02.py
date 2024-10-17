# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 1: Fusion de imagenes mediante piramides
# Memoria: codigo de la pregunta 02

# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# PAREJA/TURNO: 01/VIERNES

from p1_pregunta_01 import cargar_imagenes, mostrar_imagen, run_fusion_color
from p1_tarea4 import run_fusion
from p1_utils import visualizar_gaus_piramide, visualizar_lapl_piramide




def experimentar_fusion(path_imagenes, niveles_explorados):
    # Cargar imágenes y máscara
    imgA, imgB, mask = cargar_imagenes(path_imagenes)
    
    for niveles in niveles_explorados:
        print(f"Fusionando con {niveles} niveles en la pirámide...")
        
        # Fusión de imágenes en color
        img_fusionada = run_fusion_color(imgA, imgB, mask, niveles)

        # Mostrar la imagen fusionada
        mostrar_imagen(img_fusionada, titulo=f"Imagen Fusionada con {niveles} niveles")

        # Visualizar pirámides Gaussianas y Laplacianas si es necesario
        Gpyr_imgA, Gpyr_imgB, Gpyr_mask, Lpyr_imgA, Lpyr_imgB, Lpyr_fus, _ = run_fusion(imgA[:, :, 0], imgB[:, :, 0], mask[:, :, 0], niveles)
        visualizar_gaus_piramide(Gpyr_imgA)  
        visualizar_lapl_piramide(Lpyr_fus)  

# Ejemplo de uso
if __name__ == "__main__":
    path_imagenes = "/home/e462135/TSV/p1/img/"
    
    # Lista de números de niveles a explorar
    niveles_explorados = [1, 3, 5, 7, 10]

    # Realizar los experimentos
    experimentar_fusion(path_imagenes, niveles_explorados)

