import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
MAX_FEATURES = 42 * 2  # Maximum features based on hand landmarks

# Load the data from the pickle file
data_dict = pickle.load(open('data1111.pickle', 'rb'))
data = data_dict['data']

# Standardize lengths of feature vectors
data_fixed = []
for feature_vector in data:
    if len(feature_vector) > MAX_FEATURES:
        feature_vector = feature_vector[:MAX_FEATURES]  # Trim to max length
    elif len(feature_vector) < MAX_FEATURES:
        feature_vector.extend([0] * (MAX_FEATURES - len(feature_vector)))  # Pad with zeros
    data_fixed.append(feature_vector)

# Convert data and labels into numpy arrays
data = np.asarray(data_fixed)
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict on the test set and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

# Print the classification accuracy
print(f'{score * 100}% of samples were classified correctly!')

# Save the trained model with the new name 'model1111.p'
with open('RFC/model1111.p', 'wb') as f:
    pickle.dump({'model': model}, f)
