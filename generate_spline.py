#!/usr/bin/env python3
"""Generate a spline to avoid a set of obstacles in the 2D plane"""

from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize, basinhopping

# Create a range of fairly widely spaced Y axis values to optimize
y = np.arange(-10, 2, 1)
# Define the edges of the road in terms of linear functions of the form x=my+b (all units are in meters)
left_edge = 0.02 * y ** 2 + 0.5 * y + 5
right_edge = 0.02 * y ** 2 + 0.5 * y
# Define a set of obstacles present within the road
obstacles = [(-1.6, -5), (0.6, -2), (5.1, -1), (0.8, -1.4), ((1.5, -3))]
# Get the X and Y positions of the obstacles individually
obstacles_x, obstacles_y = [np.array(value_list) for value_list in zip(*obstacles)]
# The optimal free path (assuming no obstacles) should be the average of the left and right edges
free_path = (left_edge + right_edge) / 2


def path_loss(path):
    """Given a set of X values corresponding to the predefined Y range, return a loss that optimizes distance from obstacles as well as the ideal path"""
    # Get the mean squared error of the path from the constant free path
    path_mean_squared_error = np.mean(np.square(free_path - path))
    # Create a list containing things to add up to produce the final loss
    loss_values = [path_mean_squared_error]
    # Iterate over the path, calculating differences from one point to the next so the derivative is incorporated into the loss
    for point_index in range(len(path) - 1):
        # Add the squared difference between the derivative and the free path derivative (multiplied by a weight) to the loss list
        loss_values.append(np.square((path[point_index] - path[point_index + 1]) - (free_path[point_index] - free_path[point_index])) * 0.2)
    # Iterate over the obstacles, adding to the loss
    for obstacle_x, obstacle_y in obstacles:
        # Get the squared Pythagorean distance from each point on the path to this obstacle
        squared_distances = np.square(obstacle_x - path) + np.square(obstacle_y - y)
        # Invert all of these squared distances so closer is worse, multiply them by a constant, and add them to the loss
        loss_values.append(np.mean(1 / squared_distances) * 2)
    # Return the aggregated loss
    return np.sum(loss_values)


# Minimize the loss to produce an optimal path
path = minimize(path_loss, x0=free_path, method='TNC', jac=grad(path_loss)).x
# Fit a spline to the points (switching X and Y because Y is increasing, not X)
spline = splrep(y, path)
# Evaluate the spline on a denser Y range
dense_y = np.arange(-10, 2, 0.05)
spline_x = splev(dense_y, spline)

# Display the relevant edges and paths
plt.plot(left_edge, y, right_edge, y, spline_x, dense_y, free_path, y)
# Scatter plot the obstacles
plt.scatter(*zip(*obstacles))
# Show the combined plot on screen
plt.show()
