# Overfitting with complex models
#Author: [Alaa Othman]
#Date: [April/2025]
#Learning From Data (LFD) Course: Lect 5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed
np.random.seed(40)

# Complex target function
def true_function(x):
    return (
        8*x**7 - 15*x**6 + 10*x**5 - 2*x**4 - 3*x**3 + 1.5*x**2 - 0.5*x 
        + 0.4*np.exp(-4*x)*(3*x**4 - 2*x**3 + x)
        - 0.3*np.exp(3*x)*(x**5 - x**2)
        + 0.2*(x**3*(1-x)**2)*np.exp(5*x)
        + 0.3*np.sin(10 * np.pi * x) * (x**2 + 0.2)
        + 0.25*np.sin(20 * np.pi * x) * np.exp(-3 * (x - 0.5)**2)
        + 0.15*np.cos(12 * np.pi * x) * x * (1 - x)
    )
    #return 2*x**5 - 3*x**4 + x**3 + 2*x**2 + 0.5*np.sin(8*np.pi*x)

# Generate training data
def generate_data(n_samples=12):
    x = np.linspace(0, 1, n_samples)
    y = true_function(x) 
    return x.reshape(-1, 1), y

# Data prep
X_train, y_train = generate_data()
X_grid = np.linspace(0, 1, 1000).reshape(-1, 1)
y_true = true_function(X_grid.flatten())

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Model 1: Linear
poly1 = PolynomialFeatures(degree=1)
model1 = LinearRegression().fit(poly1.fit_transform(X_train), y_train)
y_pred1 = model1.predict(poly1.transform(X_grid))
train_mse1 = mean_squared_error(y_train, model1.predict(poly1.transform(X_train)))
test_mse1 = mean_squared_error(y_true, y_pred1)

# Model 2: Degree 7
poly7 = PolynomialFeatures(degree=7)
model7 = LinearRegression().fit(poly7.fit_transform(X_train), y_train)
y_pred7 = model7.predict(poly7.transform(X_grid))
train_mse7 = mean_squared_error(y_train, model7.predict(poly7.transform(X_train)))
test_mse7 = mean_squared_error(y_true, y_pred7)

# Model 3: Degree 15
poly15 = PolynomialFeatures(degree=15)
model15 = LinearRegression().fit(poly15.fit_transform(X_train), y_train)
y_pred15 = model15.predict(poly15.transform(X_grid))
train_mse15 = mean_squared_error(y_train, model15.predict(poly15.transform(X_train)))
test_mse15 = mean_squared_error(y_true, y_pred15)

# Combined plot
plt.figure()
plt.scatter(X_train, y_train, s=50, facecolors='none', edgecolors='k', label='Training Data')
plt.plot(X_grid, y_true, 'g--', linewidth=2, label='True Function')
plt.plot(X_grid, y_pred1, 'r', linewidth=1.5, label=f'Linear Model (MSE: {test_mse1:.3f})')
plt.plot(X_grid, y_pred7, 'orange', linewidth=1.5, label=f'Degree 7 Model (MSE: {test_mse7:.3f})')
plt.plot(X_grid, y_pred15, 'blue', linewidth=1.5, label=f'Degree 15 Model (MSE: {test_mse15:.3f})')
plt.title('Model Comparison: Underfitting vs Balanced vs Overfitting')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()