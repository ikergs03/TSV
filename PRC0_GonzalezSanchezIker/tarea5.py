import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# 1. Crear una lista vacía
lista_img = []

# 2. Leer la imagen astronaut y añadirla a la lista
img_astronaut = data.astronaut()
lista_img.append(img_astronaut)
tipo_astronaut = "color" if img_astronaut.ndim == 3 else "gris"
print("Imagen astronaut: Tipo:", tipo_astronaut)

# 3. Leer la imagen camera y añadirla a la lista
img_camera = data.camera()
lista_img.append(img_camera)
tipo_camera = "color" if img_camera.ndim == 3 else "gris"
print("Imagen camera: Tipo:", tipo_camera)

# 4. Crear un array bidimensional de ceros y añadirlo a la lista
array_zeros = np.zeros((10, 10))
lista_img.append(array_zeros)
tipo_zeros = "color" if array_zeros.ndim == 3 else "gris"
print("Array de ceros: Tipo:", tipo_zeros)

# 5. Recorrer la lista y mostrar información
for i in range(len(lista_img)):
    img = lista_img[i]
    print(f"\nElemento {i + 1}:")
    print("Dimensiones:", img.shape)
    print("Tipo:", img.dtype)
    print("Valor máximo:", np.max(img))
    
    # Visualizar la imagen
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.axis('off')
    plt.title(f'Imagen {i + 1}')
    plt.show()

# 6. Crear una nueva lista con los dos primeros elementos
nueva_lista = lista_img[:2]
print("\nNúmero de elementos en nueva_lista:", len(nueva_lista))
