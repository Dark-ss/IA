import cv2
import os
import numpy as np

dataPath = 'C:/Users/erika/OneDrive/Escritorio/IA/data'
emotionList = os.listdir(dataPath)
print('lista de emociones', emotionList)

labels = []
facesData = []
label = 0

for nameDir in emotionList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo Imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        #lee las imagenes y cuenta cada etiqueta
        facesData.append(cv2.imread(personPath + '/' +fileName, 0))
    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('ModeloFaceFrontalData2024.xml')
print("Modelo Guardado")