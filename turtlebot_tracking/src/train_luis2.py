import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def load_data_from_txt(path):
    images = []
    labels = []

    with open(path, 'r') as file:
        for line in file:
            image, label = line.strip().split()
            print(image)
            img = load_img(image, target_size=(224,224))
            img_array = img_to_array(img) /255.0
            images.append(img_array)
            labels.append(int(label))

    return np.array(images), np.array(labels)

txt_path = "train.txt"

X,y = load_data_from_txt(txt_path)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar MobileNetV2 preentrenado en ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Construir el modelo de transferencia de aprendizaje
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

model.save("modelo_luis.h5")
