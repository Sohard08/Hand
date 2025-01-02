import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Paths
DATA_DIR = 'data/archive (1)/Training Set'  # Update with your dataset path
SAVE_FILE = 'data1111.pickle'

# Dynamically Map Labels
classes = sorted(os.listdir(DATA_DIR))  # Sorted to ensure consistency
label_map = {name: idx for idx, name in enumerate(classes)}

# Feature Extraction
data = []
labels = []

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(label_map[class_name])

# Save Extracted Features
with open(SAVE_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_map': label_map}, f)

print(f"Feature extraction complete. Data saved to {SAVE_FILE}.")
