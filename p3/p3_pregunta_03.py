# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 3: Reconocimiento de escenas con modelos BOW
# Memoria - Pregunta 3.3

# AUTOR2: LÓPEZ MARTÍNEZ, ALEJANDRO
# AUTOR1: GONZÁLEZ SÁNCHEZ, IKER
# PAREJA/TURNO: 01/VIERNES

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from p3_tarea1 import construir_vocabulario, obtener_bags_of_words
from p3_utils import load_image_dataset
from p3_tarea2 import obtener_features_hog
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = 'dataset_scenes15/scenes15'
TRAIN_TEST_RATIO = 0.20
MAX_PER_CATEGORY = 200
VOCAB_SIZE = 100  # Using optimal vocab size from P3.2

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

def evaluar_random_forest(bow_train, bow_test, y_train, y_test, n_estimators=100, max_depth=None):
    # Preprocess data with StandardScaler
    scaler = StandardScaler()
    bow_train_scaled = scaler.fit_transform(bow_train)
    bow_test_scaled = scaler.transform(bow_test)

    # Create and train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                random_state=42, 
                                max_depth=max_depth)
    rf.fit(bow_train_scaled, y_train)
    
    # Calculate accuracies
    acc_train = rf.score(bow_train_scaled, y_train) * 100
    acc_test = rf.score(bow_test_scaled, y_test) * 100
    
    return acc_train, acc_test, rf

def experimento_numero_estimadores():
    print("3.3.1: Impacto del Número de Estimadores en Random Forest...")
    
    # Cargar datos y obtener características
    X_train, X_test, y_train, y_test, _ = cargar_datos(DATASET_PATH, MAX_PER_CATEGORY, TRAIN_TEST_RATIO)
    features_train_hog = obtener_features_hog(X_train, tamano=100)
    features_test_hog = obtener_features_hog(X_test, tamano=100)

    # Construir vocabulario y bolsas de palabras
    vocab_hog = construir_vocabulario(features_train_hog, vocab_size=VOCAB_SIZE)
    bow_train_hog = obtener_bags_of_words(features_train_hog, vocab_hog)
    bow_test_hog = obtener_bags_of_words(features_test_hog, vocab_hog)

    # Experimentar con diferentes números de estimadores
    n_estimators_list = [10, 50, 100, 200, 500, 750, 1000, 1500, 2000]
    results_train = []
    results_test = []

    for n_estimators in n_estimators_list:
        acc_train, acc_test, modelo = evaluar_random_forest(
            bow_train_hog, bow_test_hog, y_train, y_test, n_estimators
        )
        results_train.append(acc_train)
        results_test.append(acc_test)
        print(f"n_estimators={n_estimators}: Train Accuracy: {acc_train:.2f}%, Test Accuracy: {acc_test:.2f}%")

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, results_train, label='Train Accuracy', marker='o')
    plt.plot(n_estimators_list, results_test, label='Test Accuracy', marker='o')
    plt.xlabel("Número de Estimadores")
    plt.xticks(n_estimators_list)
    plt.ylabel("Precisión (%)")
    plt.title("Impacto del Número de Estimadores en Random Forest")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Encontrar el mejor número de estimadores
    mejor_indice = results_test.index(max(results_test))
    mejor_n_estimadores = n_estimators_list[mejor_indice]
    
    print(f"\nMejor configuración:")
    print(f"Número de estimadores: {mejor_n_estimadores}")
    print(f"Precisión de entrenamiento: {results_train[mejor_indice]:.2f}%")
    print(f"Precisión de test: {results_test[mejor_indice]:.2f}%")

    return mejor_n_estimadores

if __name__ == "__main__":
    experimento_numero_estimadores()