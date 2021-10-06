# Importamos las librerías necesarias

import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# Cargamos el modelo que creamos y entrenamos previamente

model = load_model("best_model.h5")

# Cargamos el clasificador facial del paquete 'haar cascade'

face_haar_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# Definimos el dispositivo de captura de video

cap = cv2.VideoCapture(1)

while True:

    #Capturamos frames mediante el dispositivo de captura con la biblioteca CV

    ret, test_img = cap.read()

    if not ret:
        continue

    #Convertimos la imagen que obtuvimos de color a una escala de grices

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    #Utilizamos la detección facial de 'haar cascade' para detectar rostros faciales en la imagen

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    #Iteramos los resultados de los rostros encontrados anteriormente

    for (x, y, w, h) in faces_detected:

        #Modificamos la imagen del rostro obtenido a medidas estandar

        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        #Pasamos la imagen anterior al modelo para que realice la predicción emocional

        predictions = model.predict(img_pixels)

        #Obtenemos el índice de la emoción promedio mayor

        max_index = np.argmax(predictions[0])

        #Obtenemos la emoción promedio mayor

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        #Insertamos un label que indique la emoción obtenida en el paso anterior

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #Modificamos el tamaño de la imagen del frame obtenido para mostrarla

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    #Agregamos un listener que cerrará la ejecución del programa cuando se oprima la tecla 'q'

    if cv2.waitKey(10) == ord('q'):
        break   

#Cerramos el dispositivo de captura de video y cerramos la ventana de la ejecución

cap.release()
cv2.destroyAllWindows