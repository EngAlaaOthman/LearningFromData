# Regression Numerical example
import numpy as np
import matplotlib.pyplot as plt

# Given data points
X = np.array([1, 2, 3, 4, 5, 6])  # Feature values
y = np.array([1, 3, 4, 2, 5, 5])  # Target values

# Step 1: Construct the design matrix (adding a bias term)
X_design = np.column_stack((np.ones(len(X)), X))  # Adding a column of ones

# Step 2: Compute the Normal Equation: w = (X^T X)^(-1) X^T y
XTX = np.dot(X_design.T, X_design)  # Compute X^T * X
XTy = np.dot(X_design.T, y)         # Compute X^T * y
w = np.linalg.inv(XTX).dot(XTy)     # Compute w = (X^T X)^(-1) * X^T y

# Extracting the computed values of w0 and w1
w0, w1 = w
print(f"Computed weights: w0 = {w0:.4f}, w1 = {w1:.4f}")

# Step 3: Plot the data points and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')  # Scatter plot of data points
plt.plot(X, w0 + w1 * X, color='red', linestyle='-', label=f'Best-fit line: y = {w0:.2f} + {w1:.2f}x')  # Regression line

# Labels and title
plt.xlabel("Feature (x)")
plt.ylabel("Target (y)")
plt.title("Linear Regression using Normal Equation")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()