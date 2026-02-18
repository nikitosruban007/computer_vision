import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("focus.monitor.app")

import os
import queue
import threading
import time
import numpy as np
import cv2
import mediapipe as mp
import torch
import pygame
import tkinter as tk
from ultralytics import YOLO
from winotify import Notification, audio
import pystray
from PIL import Image, ImageDraw

event_queue = queue.Queue()

def notify(text):
    event_queue.put(("notify", text))

pygame.mixer.init(frequency=44100, size=-16, channels=2)
pygame.mixer.set_num_channels(1)
channel = pygame.mixer.Channel(0)

def create_beep(freq=650, duration=0.4):
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    audio = (tone * 32767).astype(np.int16)
    stereo = np.column_stack((audio, audio))
    return pygame.sndarray.make_sound(stereo)

beep = create_beep()

settings = {
    "warn_time": 5,
    "sound_time": 7,
    "sleepy_time": 1.2,
    "ear_threshold": 0.19
}

current_state = "starting"
phone_memory_until = 0
running = True

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt").to(device)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def cv_loop():
    global current_state, phone_memory_until, running

    cap = cv2.VideoCapture(0)

    state = "focused"
    state_start = time.time()
    eyes_closed_since = None
    notified = False

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        yolo_res = yolo(frame, verbose=False, device=device)[0]
        for box in yolo_res.boxes:
            if yolo.names[int(box.cls[0])] == "cell phone" and float(box.conf[0]) > 0.5:
                phone_memory_until = time.time() + 3

        phone = time.time() < phone_memory_until

        if not result.multi_face_landmarks:
            new_state = "absent"
        else:
            lm = result.multi_face_landmarks[0].landmark

            def ear(indices):
                pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in indices]
                A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                return (A+B)/(2.0*C)

            avg_ear = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2
            nose = lm[1]
            chin = lm[152]
            dy = chin.y - nose.y
            dx = lm[454].x - lm[234].x

            now = time.time()

            if avg_ear < settings["ear_threshold"]:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                closed = now - eyes_closed_since
            else:
                eyes_closed_since = None
                closed = 0

            if phone:
                new_state = "phone"
            elif closed > settings["sleepy_time"]:
                new_state = "sleepy"
            elif dy > 0.23:
                new_state = "looking_down"
            elif abs(dx) < 0.27:
                new_state = "focused"
            else:
                new_state = "distracted"

        if new_state != state:
            state = new_state
            current_state = state
            state_start = time.time()
            notified = False

        unfocused_time = time.time() - state_start

        if state != "focused":
            if unfocused_time > settings["warn_time"] and not notified:
                notify("Повернись до роботи")
                notified = True

            if unfocused_time > settings["sound_time"]:
                volume = min((unfocused_time - settings["sound_time"]) / 6, 1.0)
                channel.set_volume(volume)
                if not channel.get_busy():
                    channel.play(beep)
        else:
            channel.stop()

def create_icon():
    img = Image.new('RGB', (64, 64), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=(0, 200, 0))
    return img

def show_ui(icon, item):
    root.after(0, root.deiconify)

def quit_app(icon, item):
    global running
    running = False
    icon.stop()
    root.destroy()
    os._exit(0)

root = tk.Tk()
root.title("Focus Monitor")
root.geometry("380x260")

def minimize_to_tray():
    root.withdraw()

root.protocol("WM_DELETE_WINDOW", minimize_to_tray)

status_var = tk.StringVar()
status_var.set("starting")

tk.Label(root, text="Current State:", font=("Arial",12)).pack(pady=5)
tk.Label(root, textvariable=status_var, font=("Arial",16,"bold")).pack()

def update_status():
    status_var.set(current_state.upper())
    root.after(500, update_status)

update_status()

def test_notify():
    notify("Тестове повідомлення")

def test_sound():
    channel.set_volume(1)
    channel.play(beep)

tk.Button(root, text="Test Notification", command=test_notify).pack(pady=5)
tk.Button(root, text="Test Sound", command=test_sound).pack(pady=5)

def process_events():
    while not event_queue.empty():
        event, data = event_queue.get()

        if event == "notify":
            toast = Notification(
                app_id="Focus Monitor",
                title="Focus Monitor",
                msg=data,
                duration="short"
            )
            toast.set_audio(audio.Default, loop=False)
            toast.show()

    root.after(200, process_events)

process_events()

threading.Thread(target=cv_loop, daemon=True).start()

icon = pystray.Icon(
    "FocusMonitor",
    create_icon(),
    "Focus Monitor",
    menu=pystray.Menu(
        pystray.MenuItem("Open Settings", show_ui),
        pystray.MenuItem("Quit", quit_app)
    )
)

threading.Thread(target=icon.run, daemon=True).start()

root.mainloop()
