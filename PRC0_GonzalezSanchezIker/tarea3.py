import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# 1. Leer y visualizar la imagen RGB
url = 'https://bit.ly/2Zjkcm7'
image_rgb = io.imread(url)

plt.imshow(image_rgb)
plt.axis('off')  # Ocultar ejes
plt.title('Imagen Original: A Small Cup of Coffee')
plt.show()

# 2. Mostrar dimensiones, tipo y valor máximo de la imagen
print("Dimensiones de la imagen:", image_rgb.shape)
print("Tipo de la imagen:", image_rgb.dtype)
print("Valor máximo de la imagen:", np.max(image_rgb))

# 3. Cambiar tipo a float y normalizar la imagen
image_float = image_rgb.astype(float) / 255.0
print("Dimensiones de la imagen normalizada:", image_float.shape)
print("Tipo de la imagen normalizada:", image_float.dtype)

# 4. Transformar la imagen al espacio de color HSV
image_hsv = color.rgb2hsv(image_float)

# 5. Visualizar cada canal en una sola ventana
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Mostrar cada canal
channels = ['Hue', 'Saturation', 'Value']
for i in range(3):
    axs[i].imshow(image_hsv[..., i], cmap='gray')
    axs[i].set_title(channels[i])
    axs[i].axis('off')  

plt.show()
