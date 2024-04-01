import numpy as np
import pyswarms as ps


# Define the objective function to be optimized (e.g., Sphere function)
def sphere(x):
    return np.sum(x ** 2)


# Set up parameters for PSO
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.8}

# Initialize the optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

# Perform optimization with adaptive inertia weight
best_position, best_fitness = optimizer.optimize(sphere, iters=100)

print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
