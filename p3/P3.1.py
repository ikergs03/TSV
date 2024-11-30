from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from p3_tarea1 import construir_vocabulario, obtener_bags_of_words
from p3_utils import load_image_dataset
from p3_tarea2 import obtener_features_tiny, obtener_features_hog
import matplotlib.pyplot as plt

DATASET_PATH = 'C:/Users/2alex/PycharmProjects/TSV/p3/dataset_scenes15/scenes15'
TRAIN_TEST_RATIO = 0.20
MAX_PER_CATEGORY = 200
VOCAB_SIZE = 50
K_NEIGHBORS = 5


# Funciones para cargar y dividir datos
def cargar_datos(dataset_path, max_per_category, train_test_ratio):
    data = load_image_dataset(container_path=dataset_path,
                              resize_shape=None,
                              max_per_category=max_per_category)
    X_train, X_test, y_train, y_test = train_test_split(data.filenames,
                                                        data.target,
                                                        test_size=train_test_ratio,
                                                        random_state=42,
                                                        stratify=data.target)
    return X_train, X_test, y_train, y_test, data.target_names


# Función para evaluar un clasificador KNN
def evaluar_clasificador(bow_train, bow_test, y_train, y_test, k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(bow_train, y_train)
    acc_train = knn.score(bow_train, y_train) * 100
    acc_test = knn.score(bow_test, y_test) * 100
    return acc_train, acc_test


# Variables globales
X_train, X_test, y_train, y_test, target_names = cargar_datos(DATASET_PATH, MAX_PER_CATEGORY, TRAIN_TEST_RATIO)
features_train_hog = obtener_features_hog(X_train, tamano=100)
features_test_hog = obtener_features_hog(X_test, tamano=100)
features_train_tiny = obtener_features_tiny(X_train, tamano=64)
features_test_tiny = obtener_features_tiny(X_test, tamano=64)


def experimento_tiny_vs_hog():
    print("3.1.1: Comparando Tiny y HOG...")

    # Crear vocabulario
    vocab_tiny = construir_vocabulario(features_train_tiny, vocab_size=VOCAB_SIZE)
    vocab_hog = construir_vocabulario(features_train_hog, vocab_size=VOCAB_SIZE)

    # Generar histogramas BoW
    bow_train_tiny = obtener_bags_of_words(features_train_tiny, vocab_tiny)
    bow_test_tiny = obtener_bags_of_words(features_test_tiny, vocab_tiny)

    bow_train_hog = obtener_bags_of_words(features_train_hog, vocab_hog)
    bow_test_hog = obtener_bags_of_words(features_test_hog, vocab_hog)

    # Entrenar y evaluar clasificador
    acc_train_tiny, acc_test_tiny = evaluar_clasificador(bow_train_tiny, bow_test_tiny, y_train, y_test, K_NEIGHBORS)
    acc_train_hog, acc_test_hog = evaluar_clasificador(bow_train_hog, bow_test_hog, y_train, y_test, K_NEIGHBORS)

    print(f"Tiny: Train Accuracy: {acc_train_tiny:.2f}%, Test Accuracy: {acc_test_tiny:.2f}%")
    print(f"HOG: Train Accuracy: {acc_train_hog:.2f}%, Test Accuracy: {acc_test_hog:.2f}%")


def experimento_vocabulario():
    print("3.1.2: Impacto del Tamaño del Vocabulario...")
    vocab_sizes = [10, 50, 100, 150, 200]
    results_train = []
    results_test = []

    for vocab_size in vocab_sizes:
        vocab_hog = construir_vocabulario(features_train_hog, vocab_size=vocab_size)
        bow_train_hog = obtener_bags_of_words(features_train_hog, vocab_hog)
        bow_test_hog = obtener_bags_of_words(features_test_hog, vocab_hog)
        acc_train, acc_test = evaluar_clasificador(bow_train_hog, bow_test_hog, y_train, y_test, K_NEIGHBORS)
        results_train.append(acc_train)
        results_test.append(acc_test)

    # Grafica de resultados
    plt.plot(vocab_sizes, results_train, label='Train Accuracy')
    plt.plot(vocab_sizes, results_test, label='Test Accuracy')
    plt.xlabel("Tamaño del Vocabulario")
    plt.ylabel("Precisión (%)")
    plt.title("Impacto del Tamaño del Vocabulario en el Desempeño")
    plt.legend()
    plt.grid()
    plt.show()


def experimento_numero_vecinos():
    print("3.1.3: Impacto del Número de Vecinos...")
    k_values = range(1, 22, 2)
    results_train_k = []
    results_test_k = []

    vocab_optimo = construir_vocabulario(features_train_hog, vocab_size=100)
    bow_train_hog = obtener_bags_of_words(features_train_hog, vocab_optimo)
    bow_test_hog = obtener_bags_of_words(features_test_hog, vocab_optimo)

    for k in k_values:
        acc_train, acc_test = evaluar_clasificador(bow_train_hog, bow_test_hog, y_train, y_test, k)
        results_train_k.append(acc_train)
        results_test_k.append(acc_test)

    # Grafica de resultados
    plt.plot(k_values, results_train_k, label='Train Accuracy')
    plt.plot(k_values, results_test_k, label='Test Accuracy')
    plt.xlabel("Número de Vecinos (k)")
    plt.ylabel("Precisión (%)")
    plt.title("Impacto del Número de Vecinos (k) en el Desempeño")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    experimento_tiny_vs_hog()
    experimento_vocabulario()
    experimento_numero_vecinos()
