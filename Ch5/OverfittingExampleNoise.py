# Overfitting with noise with Visualization
#Author: [Alaa Othman]
#Date: [April/2025]
#Learning From Data (LFD) Course: Lect 5

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_data(n_samples=20):
    x = np.linspace(0, 1, n_samples)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n_samples)
    return x.reshape(-1, 1), y

# Generate training data and true function grid
X_train, y_train = generate_data()
X_grid = np.linspace(0, 1, 1000).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X_grid)

# Initialize figure parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ==================================================================
# Step 1: Plot Training Data and True Function
# ==================================================================
plt.figure()
plt.scatter(X_train, y_train, s=50, facecolors='none', edgecolors='k', 
            label='Training Data')
plt.plot(X_grid, y_true, 'g--', linewidth=2, 
         label='True Function ($\sin(2\pi x)$)')
plt.title('Training Data and True Function')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
#plt.savefig('step1_data_and_truth.png', dpi=300, bbox_inches='tight')
#plt.close()

# ==================================================================
# Step 2: Add Complex Model (15th Degree Polynomial)
# ==================================================================
# Fit 15th degree polynomial
poly15 = PolynomialFeatures(degree=15)
X_poly15 = poly15.fit_transform(X_train)
model15 = LinearRegression().fit(X_poly15, y_train)

# Generate predictions and calculate MSE
y_pred15 = model15.predict(poly15.transform(X_grid))
train_mse15 = mean_squared_error(y_train, model15.predict(X_poly15))
test_mse15 = mean_squared_error(y_true, y_pred15)

# Create plot
plt.figure()
plt.scatter(X_train, y_train, s=50, facecolors='none', edgecolors='k')
plt.plot(X_grid, y_true, 'g--', linewidth=2)
plt.plot(X_grid, y_pred15, 'b', linewidth=1.5, 
         label='15th Degree Polynomial')
plt.title('Adding Complex Model')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.grid(True, alpha=0.3)

# Add MSE annotation
plt.text(0.02, 0.95, f'Complex Model (Degree 15)\nTrain MSE: {train_mse15:.4f}\nTest MSE: {test_mse15:.4f}', 
         transform=plt.gca().transAxes, color='blue')
plt.legend()
plt.show()
#plt.savefig('step2_complex_model.png', dpi=300, bbox_inches='tight')
#plt.close()

# ==================================================================
# Step 3: Add Simple Model (Quadratic)
# ==================================================================
# Fit quadratic model (degree=2)
poly2 = PolynomialFeatures(degree=3)
X_poly2 = poly2.fit_transform(X_train)
model2 = LinearRegression().fit(X_poly2, y_train)

# Generate predictions and calculate MSE
y_pred2 = model2.predict(poly2.transform(X_grid))
train_mse2 = mean_squared_error(y_train, model2.predict(X_poly2))
test_mse2 = mean_squared_error(y_true, y_pred2)

# Create final comparison plot
plt.figure()
plt.scatter(X_train, y_train, s=50, facecolors='none', edgecolors='k')
plt.plot(X_grid, y_true, 'g--', linewidth=2)
plt.plot(X_grid, y_pred15, 'b', linewidth=1.5, 
         label='15th Degree Polynomial')
plt.plot(X_grid, y_pred2, 'r', linewidth=1.5, 
         label='Quadratic Model (Degree 3)')
plt.title('Comparing Both Models')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.grid(True, alpha=0.3)

# Add MSE annotations
plt.text(0.02, 0.95, f'Complex Model\nTrain MSE: {train_mse15:.4f}\nTest MSE: {test_mse15:.4f}', 
         transform=plt.gca().transAxes, color='blue')
plt.text(0.02, 0.75, f'Simple Model\nTrain MSE: {train_mse2:.4f}\nTest MSE: {test_mse2:.4f}', 
         transform=plt.gca().transAxes, color='red')
plt.legend()
plt.show()
#plt.savefig('step3_full_comparison.png', dpi=300, bbox_inches='tight')
#plt.close()