import cv2
import numpy as np

face_net = cv2.dnn.readNetFromCaffe('data/DNN/deploy.prototxt', 'data/DNN/res10_300x300_ssd_iter_140000.caffemodel')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, X, Y) = box.astype('int')
            x, y = max(0, x), max(0, y)
            X, Y = min(w - 1, X), min(h - 1, Y)
            cv2.rectangle(frame, (x, y), (X, Y), (0, 255, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            roi_gray = gray[y:Y, x:X]
            roi_color = frame[y:Y, x:X]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(10, 10))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 10, minSize=(25,25))

            for(sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    cv2.imshow('tracking face', frame)

cap.release()
cv2.destroyAllWindows()

