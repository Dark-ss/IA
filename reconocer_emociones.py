import cv2
import os
import numpy as np
from collections import Counter

# Función para obtener la imagen correspondiente a la emoción

def emotionImage(emotion):
    if emotion == 'Feliz':
        image = cv2.imread('imagenes/feliz(1).jpeg')
    if emotion == 'Enojado':
        image = cv2.imread('imagenes/enojado(1).jpeg')
    if emotion == 'Asombrado':
        image = cv2.imread('imagenes/conmocionado(1).jpeg')
    if emotion == 'Triste':
        image = cv2.imread('imagenes/triste(1).jpeg')
    return image

# Función para diagnosticar con base en las emociones detectadas

def diagnostico_emociones(emotion):
    conteo = Counter(emotion)
    if conteo['Enojado'] > 40 or conteo['Triste'] > 40:
        return 'Posible estrés o ansiedad detectado'
    elif conteo['Feliz'] > 25:
        return 'Estado emocional positivo predominante'
    else:
        return 'Emociones mixtas, sin diagnóstico claro'

# Función para mostrar el resumen de las emociones detectadas

def resumen_emociones(emociones):
    conteo = Counter(emociones)  # Contamos cuántas veces aparece cada emoción
    emociones_unicas = ['Feliz', 'Enojado', 'Asombrado',
                        'Triste']  # Lista de emociones posibles

    print("\nResumen de emociones detectadas:")
    for emocion in emociones_unicas:
        # Mostrar el conteo de cada emoción
        print(f"{emocion}: {conteo.get(emocion, 0)}")

dataPath = 'C:/Users/erika/OneDrive/Escritorio/IA/data'
imagePaths = os.listdir(dataPath)
print('imagePath=', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('ModeloFaceFrontalData2024.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emociones_detectadas = []  # Lista para almacenar las emociones detectadas

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    nFrame = cv2.hconcat(
        [frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5),
                    1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 90:
            emotion = imagePaths[result[0]]  # Emoción reconocida
            emociones_detectadas.append(emotion)  # Guardar emoción
            cv2.putText(frame, '{}'.format(emotion), (x, y - 25),
                        2, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            image = emotionImage(emotion)
            nFrame = cv2.hconcat([frame, image])
        else:
            cv2.putText(frame, 'Desconocido', (x, y-20), 2,
                        0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            nFrame = cv2.hconcat(
                [frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    cv2.imshow('nFrame', nFrame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Generar y mostrar el reporte final
if emociones_detectadas:
    resumen_emociones(emociones_detectadas)  # Mostrar el resumen de emociones
    diagnostico = diagnostico_emociones(emociones_detectadas)  # Generar diagnóstico
    print("\nDiagnóstico final:", diagnostico)
else:
    print("No se detectaron emociones durante la grabación.")
