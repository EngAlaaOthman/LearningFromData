# This code demonstrates how the number of samples affects our understanding of the target function.

import numpy as np  # Importing the NumPy library for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting

# Define the first black box target function
def BlackBox_TF1(x):
    # Returns a value based on a normal distribution, influenced by the input x
    return np.random.normal(10 - 1. / (x + 0.1))

# Define the second black box target function
def BlackBox_TF2(x):
    # Returns the sine of x
    return np.sin(x)

# Define the third black box target function as a linear function
def BlackBox_TF3(x):
    # Returns a linear function of x (e.g., y = 2x + 1)
    return 2 * x + 1

# Prompt the user to enter the number of training points
n_samples = input("Enter number of training points:")  
print("Number of training points is: " + n_samples)  # Print the entered number of samples

# Convert the input to an integer
n_samples = int(n_samples)

# Set the random seed for reproducibility
np.random.seed(0)

# Generate exponentially spaced x values between 1 and 10
x = 10 ** np.linspace(0, 1, n_samples)

# Get the corresponding y values by applying the first target function to x
y = BlackBox_TF3(x)

# Create a figure for plotting with specified size
fig = plt.figure(figsize=(9, 3.5))

# Adjust the subplot parameters for better layout
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.85, wspace=0.05)

# Plot the training points as scatter points
plt.scatter(x, y, marker='x', c='k', s=80)  # 'x' markers in black with size 80

# Set the font size for y-axis ticks
plt.yticks(fontsize=20)

# Set the font size for x-axis ticks
plt.xticks(fontsize=20)

# Set the title of the plot
plt.title('Training points', fontsize=20)

# Display the plot
plt.show()