# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import PIL.Image
import numpy as np
from gtts import gTTS
import os
 
p = "Software/Scripts/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(0)
 
while True:
    ret, facePositioner = cap.read()
    facePositioner = cv2.flip(facePositioner, 1)
    gray = cv2.cvtColor(facePositioner, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    i = 0

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        
        vermelho = (0, 0, 255)
        i = i + 1

        cv2.rectangle(facePositioner, (240, 180), (430, 350), vermelho)
        cv2.rectangle(facePositioner, (275, 325), (385, 365), vermelho)
        cv2.line(facePositioner, (330, 0), (330, 1000), vermelho)
        cv2.line(facePositioner, (0, 230), (1000, 230), vermelho)
        cv2.putText(facePositioner, 'Vovo', (x-10, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Face Positioner', facePositioner)

    # Obtendo nossa imagem através da webCam e transformando-a preto e branco.
    _, faceMapper = cap.read()
    gray = cv2.cvtColor(faceMapper, cv2.COLOR_BGR2GRAY)
        
    # Detectando as faces em preto e branco.
    rects = detector(gray, 0)

    contPonto = 0
    
    # para cada face encontrada, encontre os pontos de interesse.
    for (i, rect) in enumerate(rects):
        # faça a predição e então transforme isso em um array do numpy.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # desenhe na imagem cada cordenada(x,y) que foi encontrado.
        for (x, y) in shape:
            contPonto = contPonto + 1
            cv2.circle(faceMapper, (x, y), 2, (0, 255, 0), -1)
            # if shape[38] == shape[40]:
            #     mytext = "Olá, mundo!"
            #     audio = gTTS(text=mytext, lang="pt-br", slow=False)
            #     audio.save("example.mp3")
            #     os.system("start example.mp3")
    
    # Mostre a imagem com os pontos de interesse.
    cv2.imshow("Face Mapper", faceMapper)

    # j = cv2.waitKey(5) & 0xFF
    # if j == 255:
    #     print(rects)   

    

    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("37", shape[37])
        print("38", shape[38])
        print("39", shape[39])
        print("40", shape[40])
        print("41", shape[41])
        print("42", shape[42])
        print(type(shape[37]))
        print("-----------------")
        # mytext = "Olá, mundo!"
        # audio = gTTS(text=mytext, lang="pt-br", slow=False)
        # audio.save("example.mp3")
        # os.system("start example.mp3")
        # cwd = os.getcwd()
        # print(cwd)

        # def etc():
        #     path = "C:/Users/Arthu/Documents/GitHub/Face-Reader"
        #     dir = os.listdir(path)
        #     for file in dir:
        #         if file == "example.mp3":
        #             os.remove(file)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break



cv2.destroyAllWindows()
cap.release()