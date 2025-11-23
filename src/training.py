from utils import get_base_train_test_data, NUM_CLASSES, CLASS_MAP
import tensorflow as tf
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import ( # pyright: ignore[reportMissingImports]
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
import numpy as np
from typing import Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "./model/"

def save_model(model,
               model_name:Optional[str]="base_model.keras",
               model_dir:Optional[str]=MODEL_DIR):
    """
    Guarda el modelo entrenado en el directorio especificado.

    Args:
        model (tf.keras.Model): Modelo entrenado a guardar.
        model_name (str): Nombre del archivo del modelo.
        model_dir (str): Directorio donde se guardará el modelo.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Directorio creado: {model_dir}")

    SAVE_PATH = os.path.join(model_dir, model_name)
    print(f"Guardando el modelo en formato nativo Keras (.keras) en: {SAVE_PATH}...")

    try:
        # 1. Guardar el modelo en el nuevo formato recomendado
        model.save(SAVE_PATH)
        print("✅ Modelo guardado exitosamente en formato .keras.")

    except NameError:
        print("⚠️ Error: La variable 'model' no está definida. Asegúrate de ejecutar el código de entrenamiento primero.")
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")

def load_model(model_path:Optional[str]=os.path.join(MODEL_DIR, "base_model.keras")) -> tf.keras.Model:
    """
    Carga un modelo guardado desde el directorio especificado.

    Args:
        model_path (str): Ruta completa al archivo del modelo guardado.

    Returns:
        tf.keras.Model: Modelo cargado.
    """
    print(f"Cargando el modelo desde: {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None
    
def create_model(input_shape):
    """
    Crea y entrena un modelo CNN básico para clasificación de imágenes.
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding="same"),
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding="same"),
        Conv2D(512, (3, 3), activation='relu', padding="same"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),

        # Capa de Salida
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    return model

def train_model(model, x_train, y_train, x_val, y_val,
                epochs:Optional[int]=20,
                batch_size:Optional[int]=32):
    """
    Entrena el modelo CNN con los datos proporcionados.

    Args:
        model (tf.keras.Model): Modelo CNN a entrenar.
        x_train (np.ndarray): Datos de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        x_val (np.ndarray): Datos de validación.
        y_val (np.ndarray): Etiquetas de validación.
        epochs (int): Número de épocas para entrenar.
        batch_size (int): Tamaño del lote para el entrenamiento.

    Returns:
        tf.keras.callbacks.History: Historial del entrenamiento.
    """
    print("\n--- Iniciando Entrenamiento de la CNN ---")

    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val) # Usa x_val e y_val como datos de prueba durante el entrenamiento
    )

    print("\n--- Entrenamiento Finalizado ---")

def evaluate_model(model, x_test, y_test, save_dir=None):
    """
    Evalúa el modelo CNN con métricas completas: pérdida, precisión,
    reporte de clasificación y matriz de confusión.

    Args:
        model (tf.keras.Model): Modelo CNN a evaluar.
        x_test (np.ndarray): Imágenes de prueba.
        y_test (np.ndarray): Etiquetas de prueba (one-hot).
        class_map (dict): Diccionario de mapeo clase -> índice.
        save_dir (str): Carpeta donde guardar la matriz de confusión. Opcional.

    Returns:
        tuple: (loss, accuracy)
    """

    print("\n--- Evaluación del Modelo con Datos de Prueba (TEST) ---")

    # 1. Evaluar rendimiento
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Pérdida (Loss) en Test: {loss:.4f}")
    print(f"Precisión (Accuracy) en Test: {accuracy:.4f}")

    # 2. Predicciones
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 3. Reporte de Clasificación
    print("\n### Informe de Clasificación (Test) ###")
    target_names = list(CLASS_MAP.keys())
    labels_present = np.unique(y_true)

    # Filtrar nombres de clases reales
    target_names_present = [target_names[i] for i in labels_present]

    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels_present,
        target_names=target_names_present
    ))

    # 4. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')

    # Guardar matriz si hay ruta
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "matriz_confusion_resultados.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Matriz de Confusión guardada en: {save_path}")

    plt.show()

    print("\n--- Evaluación Finalizada ---")

    return loss, accuracy

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = get_base_train_test_data()

    input_shape = x_train.shape[1:]

    model = create_model(input_shape)
    # train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32)
    # save_model(model)
    model = load_model()
    evaluate_model(model, x_test, y_test)
