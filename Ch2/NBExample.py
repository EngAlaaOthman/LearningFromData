import numpy as np

# 🏆 Define the Play Tennis dataset (Weather conditions and "Play Tennis" decision)
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

# 📌 Define a function to compute probability density for a given class
def pdf(class_idx, x):
    mean0 = mean[class_idx]  # Mean of feature values for the given class
    var0 = var[class_idx]    # Variance of feature values for the given class
    numerator = np.exp(-((x.reshape(1, -1) - mean0) ** 2) / (2 * var0))  # Gaussian formula numerator
    denominator = np.sqrt(2 * np.pi * var0)  # Gaussian formula denominator
    return numerator / denominator  # Return probability density

# 🎯 Extract features (first four columns) from Data
features = [row[:-1] for row in Data]

# 🌍 Map categorical values to numeric IDs for easier calculations
feature_mappings = {}  # Dictionary to store mappings for each column
for i in range(len(features[0])):
    unique_values = list(set(row[i] for row in features))  # Unique values in the column
    mapping = {value: idx for idx, value in enumerate(unique_values)}  # Create mapping (e.g., 'Sunny' → 0, 'Rain' → 1)
    feature_mappings[i] = mapping  # Store in dictionary

# 🔄 Convert categorical features into numeric representations
numerical_features = np.array([[feature_mappings[i][value] for i, value in enumerate(row)] 
                                for row in features])

# 🎯 Extract class labels (last column) from Data
class_labels = [row[-1] for row in Data]

# 🔄 Convert class labels into numerical values (e.g., "No" → 0, "Yes" → 1)
class_label_mapping = {value: idx for idx, value in enumerate(set(class_labels))}
numerical_class_labels = np.array([class_label_mapping[label] for label in class_labels])

# 📌 Print mappings for better understanding
print("\nFeature Mappings: ", feature_mappings)
print("\nClass Label Mapping: ", class_label_mapping)

# 🔎 Define a test example (new weather conditions) to predict "Play Tennis" outcome
x_test = ['Sunny', 'Hot', 'High', 'Weak']  # Example from dataset

# 🔄 Convert test example to numeric form
mapped_x_test = [feature_mappings[i][value] for i, value in enumerate(x_test)]
print("\n🔍 Mapped Test Example:", mapped_x_test)

# 🔢 Compute dataset statistics
n_samples, n_features = numerical_features.shape  # Number of rows and columns
classes = np.unique(numerical_class_labels)  # Unique class labels
n_classes = len(classes)  # Number of classes

# 📊 Compute mean, variance, and prior probabilities for each class
mean = np.zeros((n_classes, n_features), dtype=np.float64)
var = np.zeros((n_classes, n_features), dtype=np.float64)
priors = np.zeros(n_classes, dtype=np.float64)

for idx, c in enumerate(classes):
    X_c = numerical_features[numerical_class_labels == c, :]  # Select rows belonging to class `c`
    mean[idx, :] = X_c.mean(axis=0)  # Mean for each feature in class `c`
    var[idx, :] = X_c.var(axis=0)  # Variance for each feature in class `c`
    priors[idx] = X_c.shape[0] / float(n_samples)  # Prior probability (P(class))

# 🖨️ Print class statistics
print("\n📊 Mean Values for Each Class:\n", mean)
print("\n📊 Variance Values for Each Class:\n", var)
print("\n📊 Prior Probabilities:\n", priors)

# 🎯 Predict the class of x_test using Naïve Bayes
posteriors = []
for idx, c in enumerate(classes):
    prior = priors[idx]
    likelihood = np.prod(pdf(idx, np.array(mapped_x_test).reshape(1, -1)))  # Compute likelihood
    posterior = prior * likelihood  # Apply Bayes' theorem
    posteriors.append(posterior)  # Store posterior probability

# 🔢 Normalize posteriors to get probability distribution
Probabilities = np.array(posteriors) / np.sum(np.array(posteriors))
predicted_class = classes[np.argmax(Probabilities)]  # Choose the class with highest probability

# 🏆 Print Prediction
print("\n🎯 The Predicted Class for", x_test, "is:", predicted_class, "(", ["No", "Yes"][predicted_class], ")")

# ------------------------------------
# ✅ Model Evaluation (Training Accuracy)
# ------------------------------------
correct_predictions = 0

for ii in range(n_samples):
    posteriors = []
    for idx, c in enumerate(classes):
        prior = priors[idx]
        likelihood = np.prod(pdf(idx, numerical_features[ii, :].reshape(1, -1)))
        posterior = prior * likelihood
        posteriors.append(posterior)

    Probabilities = np.array(posteriors) / np.sum(np.array(posteriors))
    predicted_label = classes[np.argmax(Probabilities)]  # Predicted class

    # Check if prediction is correct
    if numerical_class_labels[ii] == predicted_label:
        correct_predictions += 1

# 🔥 Compute and print accuracy
accuracy = correct_predictions / n_samples
print("\n✅ Model Training Accuracy:", accuracy)
