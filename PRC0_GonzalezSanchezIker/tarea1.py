import numpy as np

# Mostrar la versión de numpy
print("Versión de numpy:", np.__version__)

# 1. Crear un vector de tamaño 10x1 con valores aleatorios enteros entre 0 y 10
vector = np.random.randint(0, 10, size=(10, 1))
print("Vector aleatorio:\n", vector)

# 2. Crear un array bidimensional de ceros con tamaño 10x10 y tipo entero
array_zeros = np.zeros((10, 10), dtype=int)
print("Array de ceros:\n", array_zeros)

# 3. Hacer que las posiciones de los extremos del array bidimensional tengan el valor 1
array_zeros[0, :] = 1  # Fila superior
array_zeros[-1, :] = 1  # Fila inferior
array_zeros[:, 0] = 1  # Columna izquierda
array_zeros[:, -1] = 1  # Columna derecha
print("Array con bordes de 1:\n", array_zeros)

# 4. Generar un nuevo array con los elementos en posiciones impares (filas y columnas)
array_impares = np.array(array_zeros[1::2, 1::2], copy=True)
print("Array con posiciones impares:\n", array_impares)

# 5. Generar un nuevo array con los elementos en posiciones pares (filas y columnas)
array_pares = np.array(array_zeros[::2, ::2], copy=True)
print("Array con posiciones pares:\n", array_pares)

# 6. Cambiar al valor 3 la posición definida por la columna tercera y fila segunda
array_zeros[1, 2] = 3
print("Segunda fila después de cambiar:\n", array_zeros[1])
