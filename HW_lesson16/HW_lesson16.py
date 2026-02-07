import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))] if os.path.exists(VIDEO_DIR) else []

if video_files:
    VIDEO_PATH = os.path.join(VIDEO_DIR, video_files[0])
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'name')

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f'Відео недоступне за шляхом: {VIDEO_PATH}')
    print('Будь-ласка виберіть інше відео, або додайте його при його відсутності.')
    exit()

model = YOLO('yolov8n.pt')

CONF_THRESHOLD = 0.35

RESIZE_WIDTH = 960

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break


    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]

        scale = RESIZE_WIDTH / w

        new_w = int(w * scale)
        new_h = int(h * scale)

        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)

    cat_count = 0
    dog_count = 0

    CAT_CLASS_ID = 15
    DOG_CLASS_ID = 16

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            match(cls):
                case _ if cls == CAT_CLASS_ID:
                    cat_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                case _ if cls == DOG_CLASS_ID:
                    dog_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    now = time.time()
    dt = now - prev_time
    prev_time = now

    cv2.putText(frame, f'Cats: {cat_count}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, f'Dogs: {dog_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, f'Total: {cat_count + dog_count}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)



    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
