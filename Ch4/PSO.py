import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------------------------------
# 1. Define the Objective Function
# ----------------------------------------------------
def f(x, y):
    """Objective function to minimize"""
    return 0.5 * x**2 + y**2  # Convex function with a single global minimum

# ----------------------------------------------------
# 2. Generate the Contour Plot of the Function
# ----------------------------------------------------
x, y = np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100))
z = f(x, y)  # Compute function values

# Find the global minimum of the function
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

# ----------------------------------------------------
# 3. Initialize PSO Parameters
# ----------------------------------------------------
c1 = c2 = 0.1  # Cognitive and social coefficients
w = 0.8  # Inertia weight

n_particles = 20  # Number of particles
np.random.seed(100)  # Set random seed for reproducibility

# Initialize particles randomly in the search space [0,5]x[0,5]
X = np.random.rand(2, n_particles) * 5  # Particle positions
V = np.random.randn(2, n_particles) * 0.1  # Initial velocities

# Initialize best-known positions and objectives
pbest = X.copy()  # Personal best positions
pbest_obj = f(X[0], X[1])  # Personal best fitness values
gbest = pbest[:, pbest_obj.argmin()]  # Global best position
gbest_obj = pbest_obj.min()  # Global best objective value

# ----------------------------------------------------
# 4. Define the PSO Update Function
# ----------------------------------------------------
def update():
    """Perform one iteration of the Particle Swarm Optimization (PSO) algorithm."""
    global V, X, pbest, pbest_obj, gbest, gbest_obj
    
    r1, r2 = np.random.rand(2)  # Random coefficients
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)  # Update velocity
    X = X + V  # Update positions

    # Compute the objective values for updated positions
    obj = f(X[0], X[1])

    # Update personal best positions where the new position is better
    better_mask = obj < pbest_obj
    pbest[:, better_mask] = X[:, better_mask]
    pbest_obj[better_mask] = obj[better_mask]

    # Update global best if a better solution is found
    if obj.min() < gbest_obj:
        gbest = X[:, obj.argmin()]
        gbest_obj = obj.min()

# ----------------------------------------------------
# 5. Set Up the Visualization
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
fig.set_tight_layout(True)

# Contour plot of the function
img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax, label="Function Value")

# Mark the global minimum
ax.plot(x_min, y_min, marker='x', markersize=8, color="white", label="Global Minimum")

# Contour lines
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Scatter plots for particles and best positions
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5, label="Personal Best")
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5, label="Particles")
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = ax.scatter(gbest[0], gbest[1], marker='*', s=100, color='red', alpha=0.8, label="Global Best")

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Particle Swarm Optimization (PSO)")
ax.legend()

# ----------------------------------------------------
# 6. Define the Animation Function
# ----------------------------------------------------
def animate(i):
    """Update function for animation at each iteration."""
    title = f'Iteration {i:02d}'
    update()  # Perform one step of PSO
    
    # Update plots with new positions
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])  # Update arrows
    gbest_plot.set_offsets(gbest.reshape(1, -1))
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

# ----------------------------------------------------
# 7. Run and Save the Animation
# ----------------------------------------------------
anim = FuncAnimation(fig, animate, frames=50, interval=500, blit=False, repeat=True)

# Fix for the "imagemagick unavailable" error by using 'pillow' instead
anim.save("PSO.gif", dpi=120, writer=PillowWriter())

# ----------------------------------------------------
# 8. Print Final Results
# ----------------------------------------------------
print(f"PSO found best solution at f({gbest}) = {gbest_obj}")
print(f"Global optimal at f([{x_min}, {y_min}]) = {f(x_min, y_min)}")
