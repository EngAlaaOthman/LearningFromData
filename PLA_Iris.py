# This code demonstrates the Perceptron Learning Algorithm (PLA) using the Iris dataset for binary classification.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from utilLect1 import plot

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target classes

# Convert the problem to binary classification (classifying between Setosa and Versicolor)
# Use only the first two classes for simplicity
mask = (y == 0) | (y == 1)  # Keep only Setosa (0) and Versicolor (1)
X = X[mask]
y = y[mask]
y = np.where(y == 0, 1, -1)  # Convert classes to 1 and -1

# Add bias term to the feature set
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add a column of ones for bias

# Initial weights for the model: [Bias weight, Weight for Feature 1, Weight for Feature 2, Weight for Feature 3, Weight for Feature 4]
weights = [0, 0.5, 0.5, 0.5, 0.5] 
plot(X, weights, title="Epoch: 0")  # Initial plot

# Set parameters for training
nb_epoch = 10  # Number of training epochs
l_rate = 0.2   # Learning rate

## Training loop
for epoch in range(nb_epoch):
    total_error = 0  # Initialize total error for this epoch
    
    for i in range(len(X)):
        # ---> B- Model evaluation: 
        # -------> B1- Calculate predictions based on current weights
        wTX = np.dot(X[i], weights)  # Weighted sum of inputs
        pred = 1 if wTX >= 0 else -1  # Determine prediction

        # Calculate error based on prediction
        if y[i] != pred:
            total_error += 1  # Increment error count if prediction is incorrect

        # ---> C- Model update (optimization)
        if y[i] != pred:  # If there is an error, update weights
            for j in range(len(weights)):  # Update each weight
                weights[j] += (l_rate * X[i][j] * y[i]) 

    # Calculate and print the training error rate
    error_rate = total_error / len(X)  # Calculate error rate
    print('Epoch %d: Training Error Rate: %.2f' % (epoch + 1, error_rate))

    # Plot the decision boundary after each epoch
    plot(X, weights, title="Epoch: %d" % (epoch + 1))