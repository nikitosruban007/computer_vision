import cv2
import os
from ultralytics import YOLO

CONF_THRESHOLD = 0.4

TRANSPORT_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck'
}


PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
if not video_files:
    print(f"Помилка: Відео не знайдено в папці {VIDEO_DIR}, можливо не підтримуваний формат або відео відсутнє")
    exit()

INPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, video_files[0])
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, 'detected_' + video_files[0])

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Помилка: Не вдалося відкрити відео {INPUT_VIDEO_PATH}")
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Початок обробки відео: {INPUT_VIDEO_PATH}")
print("Натисніть 'q' для виходу.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    current_counts = {name: 0 for name in TRANSPORT_CLASSES.values()}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in TRANSPORT_CLASSES:

                label_name = TRANSPORT_CLASSES[cls_id]
                current_counts[label_name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{label_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    y_pos = 40
    cv2.rectangle(frame, (10, 10), (250, 10 + len(current_counts) * 35), (0, 0, 0), -1)
    
    for name, count in current_counts.items():
        text = f"{name.capitalize()}: {count}"
        cv2.putText(frame, text, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

    out.write(frame)

    cv2.imshow('YOLO Transport Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Результат збережено у: {OUTPUT_VIDEO_PATH}")
