import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Data setup with bias term (x0=1)
X = np.array([[1, 1, 1],  # x1
             [1, 3, 3]]) # x2
y = np.array([1, -1])

# Test points with bias term
test_point1 = np.array([1, 3, 4])
test_point2 = np.array([1, 0, 1])

# Initialize weights
w = np.zeros(3)
eta = 0.5
n_iterations = 20

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Create grid for visualization
x1_min, x1_max = -1, 5
x2_min, x2_max = -1, 5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                       np.linspace(x2_min, x2_max, 200))

def plot_decision_boundary(w, iteration):
    # Create grid matrix with bias term
    grid = np.c_[np.ones(xx1.ravel().shape[0]), xx1.ravel(), xx2.ravel()]
    
    # Calculate predictions using the current model
    Z = np.sign(grid @ w)
    Z = Z.reshape(xx1.shape)

    # Plot decision boundary with region coloring
    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(('#FFAAAA', '#AAAAFF')))
    plt.contour(xx1, xx2, Z, levels=[0], linewidths=2, colors='black', label='Decision Boundary')
    
    # Plot data points
    plt.scatter(X[:,1], X[:,2], c=y, cmap=ListedColormap(('red', 'blue')), s=150, edgecolors='k', label='Data Points')
    plt.scatter(test_point1[1], test_point1[2], c='purple', marker='x', s=250, linewidths=2, label='Test Point 1')
    plt.scatter(test_point2[1], test_point2[2], c='orange', marker='X', s=250, linewidths=2, label='Test Point 2')
    
    # Formatting
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(f"Iteration {iteration}\nWeights: {w.round(4)}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Perform multiple iterations of gradient descent
for iteration in range(n_iterations):
    print(f"\n=== Iteration {iteration + 1} ===")
    print(f"Current weights: {w.round(4)}")
    
    # 1. Compute scores
    scores = X @ w
    print(f"Scores: {scores.round(4)}")
    
    # 2. Compute sigmoid activations
    sigmoid_terms = sigmoid(-y * scores)
    print(f"Sigmoid terms: {sigmoid_terms.round(4)}")
    
    # 3. Compute gradient
    gradient = -(1 / len(X)) * (X.T @ (y * sigmoid_terms))
    print(f"Gradient: {gradient.round(4)}")
    
    # 4. Update weights
    w -= eta * gradient
    print(f"Updated weights: {w.round(4)}")
    
    # 5. Predict test points
    test_score1 = test_point1 @ w
    test_prob1 = sigmoid(test_score1)
    test_score2 = test_point2 @ w
    test_prob2 = sigmoid(test_score2)
    
    print(f"Test Point 1 Probability (class -1): {test_prob1.round(4)} {'(Misclassified)' if test_prob1 > 0.5 else '(Correct)'}")
    print(f"Test Point 2 Probability (class +1): {test_prob2.round(4)} {'(Misclassified)' if test_prob2 < 0.5 else '(Correct)'}")
    
    # Visualize decision boundary every few iterations
    if iteration % 2 == 0 or iteration == n_iterations - 1:
        plot_decision_boundary(w, iteration+1)
