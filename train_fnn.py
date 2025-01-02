import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the preprocessed pickle file
data_dict = pickle.load(open('data1111.pickle', 'rb'))

# Fixing feature lengths to ensure uniformity
MAX_FEATURES = 42 * 2
data = data_dict['data']
data_fixed = []

for feature_vector in data:
    if len(feature_vector) > MAX_FEATURES:
        feature_vector = feature_vector[:MAX_FEATURES]
    elif len(feature_vector) < MAX_FEATURES:
        feature_vector.extend([0] * (MAX_FEATURES - len(feature_vector)))
    data_fixed.append(feature_vector)

data = np.asarray(data_fixed)
labels = np.asarray(data_dict['labels'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the Feedforward Neural Network (FNN) model
model = keras.Sequential([
    keras.layers.Input(shape=(MAX_FEATURES,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the Feedforward Neural Network...")
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_predict = model.predict(x_test)
y_predict_classes = np.argmax(y_predict, axis=1)
accuracy = accuracy_score(y_test, y_predict_classes)
print(f'Feedforward Neural Network accuracy: {accuracy * 100:.2f}%')

# Save the model
print("Saving the trained Feedforward Neural Network model as 'model1111_ffnn.h5'...")
model.save('FNN/model_ffnn.h5')
