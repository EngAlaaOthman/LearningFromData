import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# 🏆 Define the Play Tennis dataset
Data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# Extract features and labels
features = np.array([row[:-1] for row in Data])  # Extract feature columns
labels = np.array([row[-1] for row in Data])  # Extract target column

# Encode categorical features using LabelEncoder
label_encoders = {}  # Dictionary to store label encoders for each column
encoded_features = np.zeros_like(features, dtype=int)  # Empty array for transformed values

for col in range(features.shape[1]):
    le = LabelEncoder()
    encoded_features[:, col] = le.fit_transform(features[:, col])  # Transform feature column
    label_encoders[col] = le  # Store encoder for future transformations

# Encode labels (Yes → 1, No → 0)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train a Gaussian Naïve Bayes classifier
model = GaussianNB()
model.fit(encoded_features, encoded_labels)

# ----------------------------------------
# Test the classifier with a new sample
# ----------------------------------------
x_test = ['Sunny', 'Cool', 'High', 'Weak']  # Example input

# Convert test sample using stored encoders
mapped_x_test = np.array([label_encoders[i].transform([x_test[i]])[0] for i in range(len(x_test))]).reshape(1, -1)

# Predict class probability & label
predicted_probabilities = model.predict_proba(mapped_x_test)  # Get class probabilities
predicted_class = model.predict(mapped_x_test)[0]  # Get predicted class

# Print the results
print("\n Test Sample:", x_test)
print("\n Mapped Test Sample:", mapped_x_test.flatten())
print("\n Predicted Probabilities:", predicted_probabilities)
print("\n Predicted Class:", label_encoder.inverse_transform([predicted_class])[0])

# ------------------------------------
#  Model Evaluation (Training Accuracy)
# ------------------------------------
predictions = model.predict(encoded_features)
accuracy = np.mean(predictions == encoded_labels)  # Compute accuracy

print("\n Model Training Accuracy:", accuracy)
print("\n All Predictions:", label_encoder.inverse_transform(predictions))
