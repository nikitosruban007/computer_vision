import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def generateImage(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    match shape:
        case "circle":
            cv2.circle(img, (100, 100), 50, color, -1)
        case "square":
            cv2.rectangle(img, (50, 50), (150, 150), color, -1)
        case "triangle":
            points = np.array([[100, 40], [40, 160], [160, 160]])
            cv2.drawContours(img, [points], 0, color, -1)
    return img

X = []
y = []

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}

shapes = [
    'circle',
    'square',
    'triangle'
]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generateImage(bgr, shape)
            mean_color = cv2.mean(img)[:3]
