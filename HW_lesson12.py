from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/shapes.csv')

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']
n_classes = len(encoder.classes_)

model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X, y, epochs=200, verbose=0)

plt.plot(history.history['loss'], label="Втрати (loss)")
plt.plot(history.history['accuracy'], label="Точність (accuracy)")
plt.xlabel('Epoch (епоха)')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

test = np.array([[18, 16, 0]], dtype=np.float32)

pred = model.predict(test)
probs = pred[0]

print("Ймовірність по кожному класу:")
for class_index, (class_name, p) in enumerate(zip(encoder.classes_, probs)):
    print(f"Клас: {class_name}, ймовірність: {p * 100:.2f}%")

best_idx = int(np.argmax(probs))
best_label = encoder.inverse_transform([best_idx])[0]
best_conf = float(probs[best_idx] * 100)

print(f"\nМодель визначила, що це: {best_label}")
print(f"Ймовірність (у відсотках): {best_conf:.2f}%")