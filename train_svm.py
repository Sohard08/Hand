import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

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

# Initialize SVM model
model = SVC(kernel='linear', probability=True, verbose=True)

# Train the model
print("Training the Support Vector Machine (SVM) model...")
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Save the model
print("Saving the trained model as 'model1111_svm.p'...")
with open('SVM/model1111_svm.p', 'wb') as f:
    pickle.dump({'model': model}, f)
