import cv2 as cv
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten  # Agregar Flatten layer

# ENTRENAMIENTO
def clasificar_datos(data, labels):
    # Crear un modelo Sequential
    model = Sequential()

    # Añadir una capa Flatten para convertir las imágenes a una dimensión
    model.add(Flatten(input_shape=(224, 224, 3)))

    # Añadir capas Dense
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Convertir la lista de imágenes a un array de NumPy
    data = np.array(data)

    # Dividir datos en conjunto de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    y_train = np.array([int(label) for label in y_train])
    y_test = np.array([int(label) for label in y_test])

    print("Inicio del entrenamiento")
    # Entrenar el modelo
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    print("Fin del entrenamiento")

    # Evaluar el modelo
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Guardar el modelo
    model.save("clasificador_adam.h5")

def leer_imagenes():
    labels = []
    data = []
    filename = 'train.txt'
    
    with open(filename, 'r') as file:
        for line in file:
            image, label = line.strip().split()
            img = cv.imread(image)
            img = cv.resize(img, (224, 224))
            data.append(img)
            labels.append(int(label))
    
    data = np.array(data)
    labels = np.array(labels)
        
    return (data, labels)

if __name__ == "__main__":
    data, labels = leer_imagenes()
    clasificar_datos(data, labels)
