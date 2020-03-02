import numpy as np
import cv2

cap = cv2.VideoCapture('Testing Videos\\original.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
dim = (1280,720)
out = cv2.VideoWriter('compressed.mp4', fourcc, 30, dim)

timer = 0

ret, frame = cap.read()
while ret:
    timer += 1
    #cv2.putText(frame, 'Recording...', (10,25), 0, 1, (0,0,255))
    reSizedFrame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    out.write(reSizedFrame)

    cv2.imshow('Compressing...', reSizedFrame)

    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
