import cv2
import dlib

reconhecedor_face = cv2.face.LBPHFaceRecognizer_create()
reconhecedor_face.read('WIP')
altura, largura = 220, 220
detector = dlib.get_frontal_face_detector()

captura_camera = cv2.VideoCapture(0)

while True:
    ret, frame = captura_camera.read()

    deteccoes = detector(frame, 1)
    for face in deteccoes:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagem_face = cv2.resize(imagem_cinza[t:b, l:r], (altura, largura))
        cv2.rectangle(frame, (l, t), (r, b), (0,0,0), 2)
        id, confianca = reconhecedor_face.predict(imagem_face)
        if id == 1:
            nome='Evaldo'
        else:
            nome = 'Desconhecido'

        cv2.putText(frame, nome, (l+40, b+25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)
        cv2.putText(frame, str(confianca), (l+60, b+25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)

    cv2.ishow('Face', frame)
