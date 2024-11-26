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

# Define the Cosine Annealing learning rate function
def cosine_annealing(t, eta_initial, eta_min, T_max):
    return eta_min + 0.5 * (eta_initial - eta_min) * (1 + np.cos(np.pi * t / T_max))

# Generate time/iteration values
t = np.linspace(0, 50, 400)

# Set parameters for the cosine annealing function
eta_min = 0.0001  # Minimum learning rate
T_max = 50       # Total number of iterations/epochs

# Generate different initial learning rates between 10^-2 and 10^-4
eta_initial_values = [10**(-i) for i in range(2, 5)]  # [0.01, 0.001, 0.0001]

# Create the plot
plt.figure(figsize=(10, 5))

# Plot Cosine Annealing learning rate for different initial learning rates
for eta_initial in eta_initial_values:
    eta_values = cosine_annealing(t, eta_initial, eta_min, T_max)
    plt.plot(t, eta_values, label=r'$\eta_{initial} = 10^{-' + str(int(-np.log10(eta_initial))) + '}$')

# Title and labels
plt.title(r'Cosine Annealing Learning Rate Schedule pour Différents $\eta_{initial}$', fontsize=18)
plt.xlabel(r'Itérations ($t$)', fontsize=16)
plt.ylabel(r'Learning Rate $\eta(t)$', fontsize=16)
plt.grid(True)
plt.legend(fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('cosine_annealing_multiple_lr_initial.png', dpi=300)

# Show the plot
plt.show()
