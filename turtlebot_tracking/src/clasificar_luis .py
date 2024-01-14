import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# cargar el modelo entrenado
model = load_model('clasificador_adam.h5')
#model = load_model('modelo_luis.h5')

# inicializar webcam
cap = cv.VideoCapture(0)

while True:
    # Leer el fotograma de la webcam
    ret, frame = cap.read()

    # redimensionar la imagen a la forma esperada por el modelo (1440000 = 224 * 224 * 3)
    img = cv.resize(frame, (224, 224))  # ajustar según la forma esperada por el modelo

    # preprocesar la imagen -> normalizar
    img = img / 255.0

    # prediccion
    prediction = model.predict(np.expand_dims(img, axis=0))[0][0]

    # mostrar el resultado
    if prediction > 0.5:
        cv.putText(frame, "Luis", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    else:
        cv.putText(frame, "NO Luis", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    
    # mostrar la img en una ventana
    cv.imshow('Webcam', frame)

    # detener si tecla 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# liberar la webcam y cerrar
cap.release()
cv.destroyAllWindows()
