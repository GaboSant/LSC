
# Clasificación de Señas de Animales (LSC) usando Esqueletos y CNN

Este proyecto implementa un sistema de reconocimiento de imágenes para clasificar señas de animales (Lengua de Señas) utilizando **Python**, **MediaPipe** y **TensorFlow/Keras**.

El enfoque principal consiste en pre-procesar las imágenes para extraer únicamente el esqueleto (pose landmarks) del sujeto, eliminando el ruido del fondo y mejorando la precisión del modelo mediante una Red Neuronal Convolucional (CNN).

## 📂 Estructura del Proyecto

```text
LSC/
│
├── dataset/             # Datos de imágenes
│   ├── train/           # Imágenes para entrenamiento (patrón: Animales_[clase]_...)
│   └── test/            # Imágenes para validación/test (patrón: [clase]2_...)
│
├── model/               # Directorio para guardar modelos entrenados
│   ├── base_model.keras        # Modelo base generado por defecto
│   └── cnn_esqueletos_v1.keras # Modelo versionado específico
│
├── utils.py             # Funciones de carga de datos y extracción de esqueletos (MediaPipe)
├── training.py          # Definición de la CNN, entrenamiento y evaluación
└── main.py              # Script de inferencia para predecir nuevas imágenes
````

## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias. Se recomienda usar un entorno virtual.

```bash
pip install opencv-python numpy mediapipe tensorflow scikit-learn tqdm matplotlib seaborn
```

## 🧠 Funcionamiento

1.  **Extracción de Características (`utils.py`):** Cada imagen pasa por **MediaPipe Pose**. Se dibuja un "esqueleto" (líneas verdes sobre fondo negro) representando la postura.
2.  **Pre-procesamiento:** El esqueleto se redimensiona a `224x224` y se normaliza (valores 0-1).
3.  **Entrenamiento (`training.py`):** Una CNN entrena sobre estas imágenes de esqueletos para aprender patrones geométricos asociados a cada animal.

## 🚀 Uso del Proyecto

### 1\. Entrenamiento del Modelo

Si deseas entrenar el modelo desde cero con tus datos en `dataset/train`:

```bash
python training.py
```

*Esto generará y guardará el archivo `model/base_model.keras` y mostrará métricas de evaluación.*

### 2\. Inferencia (Predicción)

Para predecir la clase de una imagen individual utilizando el script principal:

```bash
python main.py
```

-----

## 💎 Cómo usar el modelo `cnn_esqueletos_v1.keras`

Si ya tienes un modelo entrenado específico llamado `cnn_esqueletos_v1.keras` y quieres usarlo para hacer predicciones, sigue estos pasos:

### Opción A: Modificar `main.py`

Edita el bloque `if __name__ == "__main__":` dentro del archivo `main.py` para que apunte a tu modelo específico:

```python
# main.py

if __name__ == "__main__":
    img_path = "./dataset/test/tu_imagen.png" 
    
    # CAMBIO AQUÍ: Especifica el nombre de tu modelo v1
    model_v1_path = "./model/cnn_esqueletos_v1.keras"
    
    clase, prob = predict_image_class(img_path, model_path=model_v1_path)

    print(f"\nImagen: {img_path}")
    print(f"Modelo usado: {model_v1_path}")
    print(f"Clase predicha: {clase}")
    print(f"Probabilidad: {prob:.4f}")
```

### Opción B: Importar en otro script

Puedes crear un nuevo script o usar una notebook de Python para cargar ese modelo específico importando la función desde `main.py`:

```python
from main import predict_image_class

# Ruta de tu imagen y de tu modelo específico
imagen = "./dataset/test/jirafa2_1.22.png"
modelo_v1 = "./model/cnn_esqueletos_v1.keras"

try:
    clase, confianza = predict_image_class(imagen, model_path=modelo_v1)
    print(f"El modelo v1 predice: {clase} ({confianza*100:.2f}%)")
except Exception as e:
    print(f"Error: {e}")
```

## 📊 Clases Soportadas

El sistema está entrenado para reconocer las siguientes 6 clases:

  * Conejo
  * Gallina
  * Jirafa
  * Mariposa
  * Pavo real
  * Pez

## 📝 Notas sobre el Dataset

El script `utils.py` espera una nomenclatura específica en los archivos para asignar las etiquetas automáticamente:

  * **Train:** `Animales_conejo_...png`
  * **Test:** `conejo2_...png` (Específicamente diseñado para filtrar imágenes de una segunda persona/sesión).
