import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,  # Set the global font size
    "axes.titlesize": 18,  # Set the font size for axes titles
    "axes.labelsize": 16,  # Set the font size for axes labels
    "xtick.labelsize": 14,  # Set the font size for x-tick labels
    "ytick.labelsize": 14,  # Set the font size for y-tick labels
    "legend.fontsize": 14,  # Set the font size for legend
})

# Define the Swish and ReLU activation functions
def swish(x):
    return x / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Generate x values
x = np.linspace(-10, 10, 400)

# Calculate y values for each activation function
y_swish = swish(x)
y_relu = relu(x)

# Create the plots
plt.figure(figsize=(10, 5))

# Plot Swish activation function
plt.subplot(1, 2, 1)
plt.plot(x, y_swish, label='Swish', color='blue')
plt.title(r'Swish Activation Function', fontsize=18)
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'Swish$(x)$', fontsize=16)
plt.grid(True)
plt.legend(fontsize=14)

# Plot ReLU activation function
plt.subplot(1, 2, 2)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title(r'ReLU Activation Function', fontsize=18)
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'ReLU$(x)$', fontsize=16)
plt.grid(True)
plt.legend(fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('activation_functions.png', dpi=300)

# Show the plots
plt.show()