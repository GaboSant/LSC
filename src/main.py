from utils import extract_skeleton, CLASS_MAP
from training import load_model
import numpy as np
import tensorflow as tf

# Invertimos el diccionario: índice -> nombre de clase
INDEX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

def predict_image_class(image_path, model_path="./model/base_model.keras"):
    """
    Recibe la ruta de una imagen, extrae el esqueleto y retorna
    la clase predicha y la probabilidad asociada.

    Args:
        image_path (str): ruta a la imagen original
        model_path (str): ruta del modelo .keras guardado

    Returns:
        tuple: (class_name, probability)
    """

    # 1. Cargar el modelo
    model = load_model(model_path)
    if model is None:
        return "Error: No se pudo cargar el modelo", 0.0

    # 2. Extraer el esqueleto
    skeleton = extract_skeleton(image_path)
    if skeleton is None:
        return "Error: No se pudo procesar la imagen", 0.0

    # 3. Normalizar y expandir dimensiones
    skeleton = skeleton.astype("float32") / 255.0
    skeleton = np.expand_dims(skeleton, axis=0)  # (1, 224, 224, 3)

    # 4. Realizar predicción
    probs = model.predict(skeleton)[0]  # vector de probabilidades
    pred_idx = np.argmax(probs)
    pred_class = INDEX_TO_CLASS[pred_idx]
    pred_prob = float(probs[pred_idx])

    return pred_class, pred_prob


# Uso de ejemplo:
if __name__ == "__main__":
    img_path = "./dataset/test/conejo2_6.4286.png"  # coloca aquí tu imagen
    clase, prob = predict_image_class(img_path)

    print(f"\nImagen: {img_path}")
    print(f"Clase predicha: {clase}")
    print(f"Probabilidad: {prob:.4f}")