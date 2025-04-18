# Overfitting with small data with Visualization
#Author: [Alaa Othman]
#Date: [April/2025]
#Learning From Data (LFD) Course: Lect 5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(15)

# Generate synthetic data (only 2 points)
def generate_data(n_samples=3):
    x = np.random.rand(n_samples)
    y = np.sin(2 * np.pi * x) 
    return x.reshape(-1, 1), y

# Generate training data and true function grid
X_train, y_train = generate_data()
X_grid = np.linspace(0, 1, 1000).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X_grid)

# --- Plot Training Data and True Function ---
plt.figure()
plt.scatter(X_train, y_train, s=100, facecolors='none', edgecolors='k', 
            label='Training Data', zorder=3)
plt.plot(X_grid, y_true, 'g--', linewidth=2, label='True Function ($\sin(2\pi x)$)')
plt.title('Training Data and True Function (2 Points)')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- Fit 15th Degree Polynomial Model (Overfitting) ---
poly15 = PolynomialFeatures(degree=15)
X_poly15 = poly15.fit_transform(X_train)
model15 = LinearRegression().fit(X_poly15, y_train)
y_pred15 = model15.predict(poly15.transform(X_grid))
train_mse15 = mean_squared_error(y_train, model15.predict(X_poly15))
test_mse15 = mean_squared_error(y_true, y_pred15)

# --- Fit Linear Model (Degree 1) ---
poly1 = PolynomialFeatures(degree=1)
X_poly1 = poly1.fit_transform(X_train)
model1 = LinearRegression().fit(X_poly1, y_train)
y_pred1 = model1.predict(poly1.transform(X_grid))
train_mse1 = mean_squared_error(y_train, model1.predict(X_poly1))
test_mse1 = mean_squared_error(y_true, y_pred1)

# --- Final Comparison Plot ---
plt.figure()
plt.scatter(X_train, y_train, s=100, facecolors='none', edgecolors='k', zorder=3)
plt.plot(X_grid, y_true, 'g--', linewidth=2, label='True Function')
plt.plot(X_grid, y_pred15, 'b', linewidth=1.5, label='Degree 15 (Overfit)')
plt.plot(X_grid, y_pred1, 'r--', linewidth=1.5, label='Degree 1 (Underfit)')

plt.title('Model Comparison with Limited Training Data')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.grid(True, alpha=0.3)

# Annotate MSEs
plt.text(0.02, 0.95, f'Degree 15\nTrain MSE: {train_mse15:.4f}\nTest MSE: {test_mse15:.4f}', 
         transform=plt.gca().transAxes, color='blue', verticalalignment='top')
plt.text(0.02, 0.55, f'Degree 1\nTrain MSE: {train_mse1:.4f}\nTest MSE: {test_mse1:.4f}', 
         transform=plt.gca().transAxes, color='red', verticalalignment='top')

plt.legend()
plt.show()
