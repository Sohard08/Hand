import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set a random seed for reproducibility
tf.random.set_seed(3)

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
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Convert labels to one-hot encoding for neural network compatibility
num_classes = len(set(labels))
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(MAX_FEATURES,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training the neural network model...")
history = model.fit(
    x_train, y_train_one_hot,
    epochs=20,  # Adjust the number of epochs as needed
    batch_size=32,  # Adjust batch size as needed
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("Evaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f'{test_accuracy * 100:.2f}% of samples were classified correctly!')

# Save the model
print("Saving the trained model as 'model1111_nn.h5'...")
model.save('MLP/model1111_nn.h5')
