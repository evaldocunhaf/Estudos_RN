import cv2

detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

captura_camera = cv2.VideoCapture(0)

while True:
    #capturando frame a frame
    ret, frame = captura_camera.read()
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detector_face.detectMultiScale(imagem_cinza, minSize=(100,100), minNeighbors=5)

    #desenhando o retangulo
    for (x,y,w,h) in deteccoes:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0,0,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#encerrando a captura de video
captura_camera.release()
cv2.destroyAllWindows()
