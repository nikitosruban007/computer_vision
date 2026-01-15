import cv2
import numpy as np
import os
import shutil

PROJECT_DIR = os.path.dirname(__file__)

IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

OUT_DIR = os.path.join(PROJECT_DIR, 'out')
PEOPLE_DIR = os.path.join(OUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUT_DIR, 'no_people')

for d in [IMAGES_DIR, MODELS_DIR, OUT_DIR, PEOPLE_DIR, NO_PEOPLE_DIR]:
    os.makedirs(d, exist_ok=True)

cascade_path = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print('No face detected')
    exit()

def detect_people(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    return faces

allowed_ext = ('.jpg', '.jpeg', '.png', '.bmp')

count_people = 0
count_no_people = 0

files = os.listdir(IMAGES_DIR)

for filename in files:
    if not filename.lower().endswith(allowed_ext):
        print('Skipping ' + filename)
        continue

    in_path = os.path.join(IMAGES_DIR, filename)

    img = cv2.imread(in_path)

    if img is None:
        print("Image not found " + filename)
        continue

    faces = detect_people(img)

    if len(faces) > 0:
        out_path = os.path.join(PEOPLE_DIR, filename)
        shutil.copyfile(in_path, out_path)
        count_people += 1

        frame = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_path = os.path.join(OUT_DIR, "boxed_" + filename)
        cv2.imwrite(frame_path, frame)

    else:
        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        shutil.copyfile(in_path, out_path)
        count_no_people += 1

print('People detected: ' + str(count_people))
print('No people detected: ' + str(count_no_people))