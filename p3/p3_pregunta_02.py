# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW
# Memoria - Pregunta 3.2

# AUTOR1: APELLIDO1 APELLIDO1, NOMBRE1
# AUTOR2: APELLIDO2 APELLIDO2, NOMBRE2
# PAREJA/TURNO: NUMERO_PAREJA/NUMERO_TURNO

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from p3_tarea1 import construir_vocabulario, obtener_bags_of_words
from p3_utils import load_image_dataset
from p3_tarea2 import obtener_features_hog
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = 'dataset_scenes15/scenes15'
TRAIN_TEST_RATIO = 0.20
MAX_PER_CATEGORY = 200
VOCAB_SIZE = 50
HOG_PARAM = 100
MAX_ITER = 100000  # Aumentar el número máximo de iteraciones

def cargar_datos(dataset_path, max_per_category, train_test_ratio):
    data = load_image_dataset(container_path=dataset_path,
                              resize_shape=None,
                              max_per_category=max_per_category)
    X_train, X_test, y_train, y_test = train_test_split(data.filenames,
                                                        data.target,
                                                        test_size=train_test_ratio,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    # Cargar datos
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = cargar_datos(DATASET_PATH, MAX_PER_CATEGORY, TRAIN_TEST_RATIO)

    # Obtener características HOG
    print("Obteniendo características HOG...")
    X_train_hog = obtener_features_hog(X_train, HOG_PARAM)
    X_test_hog = obtener_features_hog(X_test, HOG_PARAM)

    # Construir vocabulario BOW
    print("Construyendo vocabulario BOW...")
    vocabulario = construir_vocabulario(X_train_hog, VOCAB_SIZE)

    # Obtener bolsas de palabras
    print("Obteniendo bolsas de palabras...")
    X_train_bow = obtener_bags_of_words(X_train_hog, vocabulario)
    X_test_bow = obtener_bags_of_words(X_test_hog, vocabulario)

    # Crear y entrenar el clasificador SVM lineal
    print("Creando y entrenando el clasificador SVM lineal...")
    svm_clf = make_pipeline(StandardScaler(), SVC(kernel='linear', max_iter=MAX_ITER))
    svm_clf.fit(X_train_bow, y_train)

    # Evaluar el clasificador
    print("Evaluando el clasificador...")
    y_train_pred = svm_clf.predict(X_train_bow)
    y_test_pred = svm_clf.predict(X_test_bow)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')

    # 3.2.1 Variar el tamaño del diccionario BOW y estudiar el rendimiento
    print("3.2.1: Variando el tamaño del diccionario BOW y estudiando el rendimiento...")
    vocab_sizes = [50, 100, 150, 200]
    train_accuracies = []
    test_accuracies = []

    for size in vocab_sizes:
        print(f"Procesando vocabulario de tamaño: {size}")
        vocabulario = construir_vocabulario(X_train_hog, size)
        X_train_bow = obtener_bags_of_words(X_train_hog, vocabulario)
        X_test_bow = obtener_bags_of_words(X_test_hog, vocabulario)
        
        svm_clf.fit(X_train_bow, y_train)
        y_train_pred = svm_clf.predict(X_train_bow)
        y_test_pred = svm_clf.predict(X_test_bow)
        
        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

    plt.plot(vocab_sizes, train_accuracies, label='Train Accuracy')
    plt.plot(vocab_sizes, test_accuracies, label='Test Accuracy')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 3.2.2 Investigar y comparar distintos tipos de kernels no lineales
    print("3.2.2: Investigando y comparando distintos tipos de kernels no lineales...")
    kernels = ['rbf', 'poly']
    for kernel in kernels:
        print(f"Procesando kernel: {kernel}")
        svm_clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, max_iter=MAX_ITER))
        svm_clf.fit(X_train_bow, y_train)
        
        y_train_pred = svm_clf.predict(X_train_bow)
        y_test_pred = svm_clf.predict(X_test_bow)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f'Kernel: {kernel}')
        print(f'Train Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}')

if __name__ == "__main__":
    main()