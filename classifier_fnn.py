import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained Feedforward Neural Network model
print("Loading the trained Feedforward Neural Network model 'model1111_ffnn.h5'...")
model = tf.keras.models.load_model('FNN/model_ffnn.h5')

# Set up MediaPipe for hand detection
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Label dictionary to map prediction to letter
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    23: 'X', 24: 'Y', 25: 'Z', 26: ' '
}

# Maximum features that will be used for prediction
MAX_FEATURES = 42 * 2

while True:
    ret, frame = cap.read()

    # Convert frame to RGB for hand tracking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    # Data to hold feature vectors for multiple hands
    predictions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Initialize the feature vector for each hand
            x_ = []
            y_ = []
            data_aux = []

            # Collect the hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Standardize feature vector length
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize based on min x
                data_aux.append(y - min(y_))  # Normalize based on min y

            # Ensure the feature vector has the correct length
            if len(data_aux) > MAX_FEATURES:
                data_aux = data_aux[:MAX_FEATURES]
            elif len(data_aux) < MAX_FEATURES:
                data_aux.extend([0] * (MAX_FEATURES - len(data_aux)))

            # Predict the character using the trained neural network
            prediction = model.predict(np.asarray([data_aux]))
            predicted_character = labels_dict[np.argmax(prediction)]
            predictions.append(predicted_character)

            # Draw landmarks and prediction on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

        # If there are multiple hands, display each prediction
        for i, predicted_character in enumerate(predictions):
            cv2.putText(frame, f'Hand {i+1}: {predicted_character}', (10, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
