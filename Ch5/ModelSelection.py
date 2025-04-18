import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(15)

# Initialize error storage arrays
sample_sizes = np.arange(2, 101, 5)
train_mse_simple, test_mse_simple = [], []
train_mse_m, test_mse_m = [], []
train_mse_complex, test_mse_complex = [], []

# Generate true function grid
X_grid = np.linspace(0, 1, 1000).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X_grid)

# Pre-generate all possible data points
X_full = np.random.rand(100)
#y_full = np.sin(2 * np.pi * X_full)+ np.random.normal(0, 0.1, 100)
#y_full = np.sin(2 * np.pi * X_full)
y_full = np.sin(2 * np.pi * X_full)+ np.random.normal(0, 0.5, 100)

def generate_data(n_samples):
    return X_full[:n_samples].reshape(-1, 1), y_full[:n_samples]

# Main computation loop
for n in sample_sizes:
    X_train, y_train = generate_data(n)
    
    # Simple model
    poly1 = PolynomialFeatures(degree=1)
    X_poly1 = poly1.fit_transform(X_train)
    model1 = LinearRegression().fit(X_poly1, y_train)
    train_mse_simple.append(mean_squared_error(y_train, model1.predict(X_poly1)))
    test_mse_simple.append(mean_squared_error(y_true, model1.predict(poly1.transform(X_grid))))
    
    # moderate model
    polym = PolynomialFeatures(degree=6)
    X_polym = polym.fit_transform(X_train)
    modelm = LinearRegression().fit(X_polym, y_train)
    train_mse_m.append(mean_squared_error(y_train, modelm.predict(X_polym)))
    test_mse_m.append(mean_squared_error(y_true, modelm.predict(polym.transform(X_grid))))
    
    # Complex model
    poly15 = PolynomialFeatures(degree=15)
    X_poly15 = poly15.fit_transform(X_train)
    model15 = LinearRegression().fit(X_poly15, y_train)
    train_mse_complex.append(mean_squared_error(y_train, model15.predict(X_poly15)))
    test_mse_complex.append(mean_squared_error(y_true, model15.predict(poly15.transform(X_grid))))

# Custom styling
plt.style.use('ggplot')  # Use a default style that is available
plt.rcParams['font.size'] = 12
colors = ['#2c7bb6', '#d7191c']
fill_alphas = [0.15, 0.1]

def create_separate_plot(train_errors, test_errors, model_name, color, fill_alpha):
    """Create a standalone plot with enhanced features"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main error curves
    ax.plot(sample_sizes, train_errors, color=color, linestyle='--', 
            marker='o', markersize=6, label='Training Error')
    ax.plot(sample_sizes, test_errors, color=color, linestyle='-', 
            linewidth=2.5, label='Test Error')
    
    # Error region filling
    ax.fill_between(sample_sizes, train_errors, test_errors, 
                    color=color, alpha=fill_alpha, label='Error Gap')
    
    # Annotations and labels
    ax.set_title(f'{model_name} Model Error Progression\nBias-Variance Tradeoff', pad=15)
    ax.set_xlabel('Number of Training Points', labelpad=10)
    ax.set_ylabel('Mean Squared Error (log scale)', labelpad=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4)
    
    # Custom legend
    legend = ax.legend(frameon=True, shadow=True, loc='upper right',
                      bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    # Annotation positioning
    if "Complex" in model_name:
        ax.annotate('Overfitting Region\n(High Variance)', 
                    xy=(10, 1e2), xytext=(20, 1e3),
                    arrowprops=dict(arrowstyle="->", color='dimgray', lw=1),
                    fontsize=10, ha='center')
        ax.annotate('Sufficient Data\nReduces Variance', 
                    xy=(80, 1e-3), xytext=(65, 1e-4),
                    arrowprops=dict(arrowstyle="->", color='dimgray', lw=1),
                    fontsize=10, ha='center')
    else:
        ax.annotate('Underfitting Region\n(High Bias)', 
                    xy=(40, 1e-1), xytext=(30, 1e-2),
                    arrowprops=dict(arrowstyle="->", color='dimgray', lw=1),
                    fontsize=10, ha='center')
    
    # Axis polishing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#808080')
    ax.spines['bottom'].set_color('#808080')
    
    plt.tight_layout()
    return fig

# Generate and display simple model plot
fig_simple = create_separate_plot(train_mse_simple, test_mse_simple, 
                                'Simple (Degree 1)', colors[0], fill_alphas[0])
plt.show()

# Generate and display simple model plot
fig_m = create_separate_plot(train_mse_m, test_mse_m, 
                                'Moderate (Degree 4)', colors[0], fill_alphas[0])
plt.show()


# Generate and display complex model plot
fig_complex = create_separate_plot(train_mse_complex, test_mse_complex, 
                                 'Complex (Degree 15)', colors[1], fill_alphas[1])
plt.show()