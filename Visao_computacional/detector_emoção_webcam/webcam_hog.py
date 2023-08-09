import cv2
import dlib
import tensorflow as tf
import numpy as np

with open('arquitetura_emocoes.json', 'r') as arquivo:
    arquitetura = arquivo.read()
rede = tf.keras.models.model_from_json(arquitetura)
rede.load_weights('pesos_emocoes.hdf5')
rede.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

emocoes = ['Raiva', 'Desgosto', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpreso']

detector_facial = dlib.get_frontal_face_detector()

captura_camera = cv2.VideoCapture(0)

while True:
    ret, frame = captura_camera.read()

    deteccoes = detector_facial(frame, 1)

    for face in deteccoes:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 0), 2)
        roi = frame[t:b, l:r]
        roi = cv2.resize(roi, (48,48))
        roi = roi / 255
        roi = np.expand_dims(roi, axis=0)
        previsao = rede.predict(roi)
        if previsao is not None:
            resultado = np.argmax(previsao)
            cv2.putText(frame, emocoes[resultado], (l+40,b+25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0),1,cv2.LINE_AA)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
captura_camera.release()
cv2.destroyAllWindows()
