# This code is to show how the relation of the number of samples and the knowledge 
# that we can get about the target function
import numpy as np
import matplotlib.pyplot as plt

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

# this to draw the target function
n_samples = 1000 
np.random.seed(0)
x = 10 ** np.linspace(0, 1, n_samples)

sampledpoints=10 ** np.linspace(0, 1, 10)
noise = np.random.normal(0,0.2,10)

# nonlinear TF
y = BlackBox_TF3(x)
y_withoutnoise = BlackBox_TF2(sampledpoints)
y_noisy = BlackBox_TF3(sampledpoints) + noise

fig = plt.figure(figsize=(9, 3.5))
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
plt.plot(x, y, c='k')
plt.scatter(sampledpoints, y_noisy, c='r', marker='x',s=160)
plt.scatter(sampledpoints, y_withoutnoise, c='b', marker='o', s=160)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(min(x)-0.1,max(x)+0.1)
plt.ylim(min(np.r_[y.reshape(-1,1),y_noisy.reshape(-1,1)])-0.1,max(np.r_[y.reshape(-1,1),y_noisy.reshape(-1,1)])+0.1)
plt.show()