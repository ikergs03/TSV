import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.ndimage import sobel, gaussian_filter

# 1. Leer y visualizar la imagen en escala de grises "brick"
image_gray = data.brick()

plt.imshow(image_gray, cmap='gray')
plt.axis('off')
plt.title('Imagen en Escala de Grises: Brick')
plt.savefig('imagen_brick.png', bbox_inches='tight')  # Guardar la imagen
plt.close()

# 2. Mostrar dimensiones de la imagen y tipo
print("Dimensiones de la imagen:", image_gray.shape)
print("Tipo de la imagen:", image_gray.dtype)

# 3. Filtrado de Sobel horizontal
sobel_horizontal = sobel(image_gray, axis=1)  # Filtrado en dirección horizontal

# Filtrado Gaussiano con sigma=10
gaussian_filtered = gaussian_filter(image_gray, sigma=10)

# 4. Visualizar la imagen original y cada resultado en ventanas distintas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Imagen original
axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Imagen Original')
axs[0].axis('off')

# Filtrado Sobel horizontal
axs[1].imshow(sobel_horizontal, cmap='gray')
axs[1].set_title('Filtrado Sobel Horizontal')
axs[1].axis('off')

# Filtrado Gaussiano
axs[2].imshow(gaussian_filtered, cmap='gray')
axs[2].set_title('Filtrado Gaussiano (sigma=10)')
axs[2].axis('off')

plt.savefig('resultados_tarea4.png', bbox_inches='tight')  # Guardar la imagen
plt.show()  # Mostrar las imágenes
