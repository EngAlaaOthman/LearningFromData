import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------
# 🏆 Load and Prepare the Dataset
# ------------------------------
iris = datasets.load_iris()
X_data = iris.data  # Features (Continuous)
y_data = iris.target  # Labels (0, 1, or 2)

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# ----------------------------------
# 📌 Compute Mean, Variance, and Priors for Each Class
# ----------------------------------
n_samples, n_features = X_train.shape  # Get dimensions
classes = np.unique(y_train)  # Get unique class labels
n_classes = len(classes)  # Number of classes

# Initialize storage for mean, variance, and prior probabilities
mean = np.zeros((n_classes, n_features), dtype=np.float64)
var = np.zeros((n_classes, n_features), dtype=np.float64)
priors = np.zeros(n_classes, dtype=np.float64)

for idx, c in enumerate(classes):
    X_c = X_train[y_train == c]  # Select rows belonging to class `c`
    mean[idx, :] = X_c.mean(axis=0)  # Compute mean for each feature
    var[idx, :] = X_c.var(axis=0)  # Compute variance for each feature
    priors[idx] = X_c.shape[0] / float(n_samples)  # Compute prior P(class)

# ----------------------------------
# 📌 Define Probability Density Function (PDF)
# ----------------------------------
def pdf(class_idx, x):
    """
    Computes the probability density function (PDF) for a given feature value x 
    using the class-specific mean and variance.
    """
    mean0 = mean[class_idx]  # Get mean of class
    var0 = var[class_idx]  # Get variance of class
    numerator = np.exp(-((x - mean0) ** 2) / (2 * var0))  # Gaussian formula numerator
    denominator = np.sqrt(2 * np.pi * var0)  # Gaussian formula denominator
    return numerator / denominator  # Return probability density

# ----------------------------------
# 📌 Naïve Bayes Prediction Function
# ----------------------------------
def predict(x_sample):
    """
    Predicts the class of a given sample using the Naïve Bayes rule.
    """
    posteriors = []
    
    for idx, c in enumerate(classes):
        prior = np.log(priors[idx])  # Log(P(class)) to prevent numerical underflow
        likelihood = np.sum(np.log(pdf(idx, x_sample)))  # Compute log-likelihood
        posterior = prior + likelihood  # Bayes Rule: log(P(class)) + sum(log(P(feature|class)))
        posteriors.append(posterior)
    
    return classes[np.argmax(posteriors)]  # Return class with highest probability

# ----------------------------------
# 📌 Model Evaluation (Training & Testing)
# ----------------------------------
# Predict on the test set
y_pred = np.array([predict(x) for x in X_test])

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy:.2f}")

# Example Prediction:
sample = X_test[0]  # Take one test instance
predicted_class = predict(sample)
print(f"\nPredicted Class for Sample {sample}: {predicted_class} (Actual: {y_test[0]})")
