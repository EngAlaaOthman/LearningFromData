# This code is to show how the relation of the number of samples and the knowledge 
# that we can get about the target function
import numpy as np
import matplotlib.pyplot as plt

def BlackBox_TF2(x):
    return np.sin(x)

def BlackBox_TF3(x):
    return 2*x

# this to draw the target function
n_samples = 1000 
np.random.seed(0)
x = 10 ** np.linspace(0, 1, n_samples)

sampledpoints=10 ** np.linspace(0, 1, 10)
noise = np.random.normal(0,0.2,10)
# linear TF
#y = BlackBox_TF3(x)
#y_withoutnoise = BlackBox_TF3(sampledpoints)
#y_noisy = BlackBox_TF3(sampledpoints) + noise

# nonlinear TF
y = BlackBox_TF2(x)
y_withoutnoise = BlackBox_TF2(sampledpoints)
y_noisy = BlackBox_TF2(sampledpoints) + noise

fig = plt.figure(figsize=(9, 3.5))
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
#plt.scatter(x, y, marker='x', c='k', s=80)
plt.plot(x, y, c='k')
plt.scatter(sampledpoints, y_noisy, c='r', marker='x',s=160)
plt.scatter(sampledpoints, y_withoutnoise, c='b', marker='o', s=160)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlim(min(x)-0.1,max(x)+0.1)
plt.ylim(min(np.r_[y.reshape(-1,1),y_noisy.reshape(-1,1)])-0.1,max(np.r_[y.reshape(-1,1),y_noisy.reshape(-1,1)])+0.1)
plt.show()