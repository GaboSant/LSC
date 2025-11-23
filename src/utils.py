import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Mapeo de clases
CLASS_MAP = {
    "conejo": 0,
    "gallina": 1,
    "jirafa": 2,
    "mariposa": 3,
    "pavo real": 4,
    "pez": 5
}
NUM_CLASSES = len(CLASS_MAP)
BASE_PATH = "./dataset/"

def extract_skeleton(image_path):
    """
    Carga una imagen y utiliza MediaPipe Pose para extraer y dibujar el esqueleto.

    Args:
        image_path (str): Ruta completa al archivo de imagen.

    Returns:
        np.ndarray or None: Imagen en formato NumPy con solo el esqueleto dibujado, o None si falla.
    """
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None

    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para obtener los landmarks
    results = pose.process(img_rgb)

    # Crear una imagen negra para dibujar solo el esqueleto
    # Usamos las dimensiones originales para el canvas negro
    skeleton_img = np.zeros_like(img)

    if results.pose_landmarks:
        # Dibujar las conexiones del esqueleto (pose connections)
        mp.solutions.drawing_utils.draw_landmarks(
            skeleton_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # Personalizar el estilo para que solo se vea el esqueleto (conexiones)
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=0), # Invisible landmarks
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1) # Conexiones verdes
        )

    # Liberar el objeto Pose
    pose.close()

    # Convertir de nuevo a BGR (formato estándar de OpenCV) para consistencia si es necesario,
    # aunque para TensorFlow es mejor mantenerlo como RGB si se desea o simplemente la imagen en escala de grises.
    # Para el entrenamiento de CNN, es común usar imágenes de 3 canales (RGB/BGR) o 1 canal (escala de grises).
    # Vamos a devolver la imagen del esqueleto en escala de grises para reducir la complejidad del modelo.
    # El esqueleto es blanco sobre negro o, en este caso, verde sobre negro.
    # Al final, simplemente devolvemos la imagen de 3 canales (BGR).
    # Opcionalmente, puedes convertir a escala de grises:
    # return cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY) # 1 canal

    # Normalizar el tamaño de la imagen para la entrada de la CNN (Ej: 224x224)
    target_size = (224, 224)
    return cv2.resize(skeleton_img, target_size)

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Asumiendo que extract_skeleton, CLASS_MAP y NUM_CLASSES ya están definidos.

def load_and_preprocess_data(base_path, class_map, target_type):
    """
    Carga las imágenes de las carpetas 'train' o 'test', extrae los esqueletos
    y prepara los datos y las etiquetas, mostrando una barra de progreso.

    Args:
        base_path (str): Ruta base hasta /content/drive/MyDrive/DF/LSC/imagenes/
        class_map (dict): Mapeo de nombre de clase a índice (ej: {"conejo": 0}).
        target_type (str): 'train' o 'test'.

    Returns:
        tuple: (np.ndarray x_data, np.ndarray y_labels)
    """
    data = []
    labels = []
    target_dir = os.path.join(base_path, target_type)

    print(f"\nIniciando carga de datos en: {target_dir}")

    # Lista todos los archivos PNG para contar el total y usar tqdm
    all_files = [f for f in os.listdir(target_dir) if f.endswith(".png")]

    # Itera sobre los archivos usando tqdm para la barra de progreso
    for filename in tqdm(all_files, desc=f"Procesando {target_type} imágenes"):
        filepath = os.path.join(target_dir, filename)

        # 1. Determinar la clase (etiqueta)
        class_name = None
        if target_type == 'train':
            # Patrón: Animales_[animal]_[t][variantes].png
            try:
                # Ejemplo: Animales_conejo_61.219_flip_rot_neg.png -> "conejo"
                part = filename.split('_')[1]
                if part in class_map:
                    class_name = part
            except IndexError:
                continue # Saltar si el nombre no sigue el patrón esperado

        elif target_type == 'test':
            # Patrón: [animal][1 o 2]_[t].png
            # Ejemplo: mariposa2_1.3222.png -> "mariposa"
            try:
                # 1. Verificar si el archivo pertenece a la Persona 2

                # El nombre del archivo debe tener el patrón: [animal]2_...
                # Buscamos el caracter que precede al primer '_'

                # Encontramos el índice donde empieza el tiempo
                if '_' not in filename:
                    continue

                first_underscore_index = filename.find('_')

                # La parte que contiene [animal] y el número de persona es antes del primer '_'
                person_id_part = filename[:first_underscore_index]

                # Verificamos si esta parte termina con '2'
                if not person_id_part.endswith('2'):
                    # Si no termina en '2', saltamos esta imagen inmediatamente.
                    continue

                # Si llegamos aquí, la imagen es de la persona 2. Ahora extraemos el nombre del animal.

                animal_name = ""
                for char in person_id_part:
                    if char.isdigit():
                        break
                    animal_name += char

                # Eliminar el número (que ya sabemos que es '2') del final del nombre
                while animal_name and animal_name[-1].isdigit():
                    animal_name = animal_name[:-1]

                if animal_name in class_map:
                    class_name = animal_name

            except Exception as e:
                # print(f"Error al parsear el archivo de test {filename}: {e}")
                continue

        if class_name and class_name in class_map:
            label = class_map[class_name]

            # 2. Extraer el esqueleto (asumiendo que `extract_skeleton` está definido)
            # NOTA: `extract_skeleton` debe estar disponible en el entorno global.
            skeleton_image = extract_skeleton(filepath)

            if skeleton_image is not None:
                # Normalizar los píxeles (0-255) a (0-1)
                normalized_image = skeleton_image.astype('float32') / 255.0
                data.append(normalized_image)
                labels.append(label)

    # Convertir a arreglos NumPy
    x_data = np.array(data)
    y_labels = np.array(labels)

    # Convertir etiquetas a formato one-hot encoding para la CNN
    # NOTA: `NUM_CLASSES` debe estar disponible en el entorno global.
    y_labels_onehot = tf.keras.utils.to_categorical(y_labels, num_classes=NUM_CLASSES)

    print(f"Carga de datos de {target_type} finalizada. Total de imágenes: {len(data)}. Forma de X: {x_data.shape}, Forma de Y: {y_labels_onehot.shape}")
    return x_data, y_labels_onehot

def get_train_test_data(base_path, class_map):
    """
    Carga todos los datos de entrenamiento y prueba.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    # Cargar datos de entrenamiento
    x_train_all, y_train_all = load_and_preprocess_data(base_path, class_map, 'train')

    # Cargar datos de prueba
    x_test, y_test = load_and_preprocess_data(base_path, class_map, 'test')

    # Dividir los datos de entrenamiento en entrenamiento real y validación
    # para usar en el entrenamiento (como pide el requerimiento)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.15, # 20% para validación
        random_state=42,
        stratify=y_train_all # Para asegurar que las clases estén bien distribuidas
    )

    # NOTA: Los requerimientos indican usar 'una parte de estos mismos' para prueba,
    # por lo que el x_val, y_val es la 'parte de prueba' durante el entrenamiento.

    return x_train, y_train, x_val, y_val, x_test, y_test

def get_base_train_test_data():
    """
    Carga todos los datos de entrenamiento y prueba, y muestra las formas finales.

    Returns:
        tuple: (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    return get_train_test_data(BASE_PATH, CLASS_MAP)

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_test_data(BASE_PATH, CLASS_MAP)

    # Comprobar las formas finales
    print("\n--- Formas Finales de los Datos ---")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape} (Usados como datos de prueba en el entrenamiento)")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape} (Usados para la evaluación final)")
    print(f"y_test shape: {y_test.shape}")