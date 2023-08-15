import cv2

rastreador = cv2.TrackerKCF_create()

video = cv2.VideoCapture('race.mp4')
ok, frame = video.read()

boudbox = cv2.selectROI(frame) #retorna 4 valores, posição X e Y de onde começa e altura  largura

ok = rastreador.init(frame, boudbox)

while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, boudbox = rastreador.update(frame)
    if ok:
        (x, y, w, h) =[int(v) for v in boudbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2, 1)
    else:
        cv2.putText(frame, 'Erro', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Rastreamento', frame)
    if cv2.waitKey(1) & 0XFF == 27: #ESC
        break