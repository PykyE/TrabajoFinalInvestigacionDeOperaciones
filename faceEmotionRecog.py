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
from time import sleep
from datetime import datetime
import _thread

# Definimos una función para calcular la emoción promedio mayor por minuto

def output():
    global minute
    global max_index
    global emotions_count
    global emotions

    # Traemos el índice de la emoción registrada en caso de que exista

    try:
        emotions_count[max_index]+=1
    except:
        pass

    # Si ha pasado un minuto, en caso positivo, se mostrará en consola la hora y la emoción mayor promedio detectada

    if datetime.now().strftime("%M") != minute:
        maxV = max(emotions_count)
        index = emotions_count.index(maxV)
        emotion = emotions[index]
        print('{0} : {1}'.format(datetime.now().strftime("%I:%M %p"), emotion))
        minute = datetime.now().strftime("%M")
        emotions_count = [0, 0, 0, 0, 0, 0, 0]
        sleep(1)
    else:
        sleep(1)

# Cargamos el modelo que creamos y entrenamos previamente

model = load_model("best_model.h5")

# Cargamos el clasificador facial del paquete 'haar cascade'

face_haar_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# Definimos el dispositivo de captura de video

cap = cv2.VideoCapture(1)

# Definimos el minuto actual en el que se inicia la ejecución del programa

minute = datetime.now().strftime("%M")

# Inicializamos los valores de las emociones registradas 

emotions_count = [0, 0, 0, 0, 0, 0, 0]

while True:

    # Capturamos frames mediante el dispositivo de captura con la biblioteca CV

    ret, test_img = cap.read()

    if not ret:
        continue

    # Convertimos la imagen que obtuvimos de color a una escala de grices

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Utilizamos la detección facial de 'haar cascade' para detectar rostros faciales en la imagen

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    # Iteramos los resultados de los rostros encontrados anteriormente

    for (x, y, w, h) in faces_detected:

        # Modificamos la imagen del rostro obtenido a medidas estandar

        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Pasamos la imagen anterior al modelo para que realice la predicción emocional

        predictions = model.predict(img_pixels)

        # Obtenemos el índice de la emoción promedio mayor

        max_index = np.argmax(predictions[0])

        # Obtenemos la emoción promedio mayor

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # Insertamos un label que indique la emoción obtenida en el paso anterior

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Modificamos el tamaño de la imagen del frame obtenido para mostrarla

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    # Llamamos a la función que definimos al comienzo para calcular la emoción promedio mayor

    _thread.start_new_thread( output, () )

    # Agregamos un listener que cerrará la ejecución del programa cuando se oprima la tecla 'q'

    if cv2.waitKey(10) == ord('q'):
        break   

#Cerramos el dispositivo de captura de video y cerramos la ventana de la ejecución

cap.release()
cv2.destroyAllWindows