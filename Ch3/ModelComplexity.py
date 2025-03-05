import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Function to generate noisy sine wave data
def generating_func(x, err=0):
    return np.random.normal(np.sin(x), err)

# Number of training samples
n_samples = 20
np.random.seed(0)  # Set random seed for reproducibility

# Generate training data (logarithmically spaced)
x_training = 10 ** np.linspace(-2, 0, n_samples)
y_training = generating_func(x_training)

# Generate test data (linearly spaced)
x_test = np.linspace(-0.2, 0, n_samples)
y_test = generating_func(x_test)

# Define different degrees of polynomial complexity
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]

# Initialize arrays to store training and test errors
Training_error = np.zeros([1, len(degrees)])
Test_error = np.zeros([1, len(degrees)])

# Loop over different polynomial degrees and fit models
for i, d in enumerate(degrees):
    # Create a polynomial regression model of degree d
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    
    # Train the model on the training data
    model.fit(x_training[:, np.newaxis], y_training)
    
    # Compute absolute error for training data
    Training_error[0, i] = np.sum(np.abs(model.predict(x_training[:, np.newaxis]) - y_training))
    
    # Compute absolute error for test data
    Test_error[0, i] = np.sum(np.abs(model.predict(x_test[:, np.newaxis]) - y_test))

# Generate index values for plotting
idx = np.arange(0, len(degrees)) + 1

# Create a plot to visualize training and test errors
fig = plt.figure(figsize=(9, 3.5))

# Plot training error (in-sample error) using log scale
plt.plot(idx, np.transpose(np.log10(Training_error)), label='In-sample error')

# Plot test error (out-of-sample error) using log scale
plt.plot(idx, np.transpose(np.log10(Test_error)), label='Out-of-sample error')

# Set axis labels and formatting
plt.xlabel('VC dimension', fontsize=20)  # Updated x-axis label
plt.ylabel('Prediction error', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(1, len(degrees))

# Add a legend to distinguish between training and test errors
plt.legend(fontsize=20, loc=3)

# Display the plot
plt.show()
