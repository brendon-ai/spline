#!/usr/bin/env python3
"""Generate a spline to avoid a set of obstacles in the 2D plane"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev

# DEFINITIONS
# Create a range of Y axis values to plot on
y = np.arange(-10, 10, 0.05)
# Define the edges of the road in terms of linear functions of the form x=my+b (all units are in meters)
left_edge = 0.02 * y ** 2 + 0.5 * y + 2
right_edge = 0.02 * y ** 2 + 0.5 * y
# Define a set of obstacles present within the road
obstacles = [(-1.5, -5), (1, -1), (5, -1), (4, 5), (4.1, 5), (1.3, -0.4)]

# COMPUTATIONS
# The optimal free path (assuming no obstacles) should be the average of the left and right edges
free_path = (left_edge + right_edge) / 2
# Initialize a list of spline points to add to the winding path
spline_points = []
# Next, we want to create a series of points that a spline can be fit to, which avoids the obstacles with the most room to spare
# Iterate over the obstacles, adding to the list of spline points
for obstacle_x, obstacle_y in obstacles:
    # Find the point on both of the road boundaries that is the closest (straight-line) to this obstacle
    left_distances = np.sqrt((left_edge - obstacle_x) ** 2 + (y - obstacle_y) ** 2)
    right_distances = np.sqrt((right_edge - obstacle_x) ** 2 + (y - obstacle_y) ** 2)
    left_closest = np.amin(left_distances)
    right_closest = np.amin(right_distances)
    left_closest_index = np.argmin(left_distances)
    right_closest_index = np.argmin(right_distances)
    # Ignore the obstacle if it doesn't come within a reasonable distance of the free path
    if np.amin([left_closest, right_closest]) > 1:
        continue
    # Choose the edge point on whichever edge is farther away, and average that and the obstacle point to get a spline point
    if left_closest > right_closest:
        spline_points.append(((left_edge[left_closest_index] + obstacle_x) / 2, (y[left_closest_index] + obstacle_y) / 2))
    else:
        spline_points.append(((right_edge[right_closest_index] + obstacle_x) / 2, (y[right_closest_index] + obstacle_y) / 2))
# Iterate over the indices of the full Y range
for y_index in range(len(y)):
    # Create a flag to determine whether this point is too close to an obstacle (assuming it is not)
    too_close = False
    # Ignore those points for which the corresponding free path points are within a predefined distance of an obstacle
    for obstacle_x, obstacle_y in obstacles:
        # Compute the distance from the obstacle to the corresponding free path point
        if np.sqrt((free_path[y_index] - obstacle_x) ** 2 + (y[y_index] - obstacle_y) ** 2) < 1:
            # If it's not too close, set the flag
            too_close = True
    # If this point isn't too close, add the free path point to the spline points
    if not too_close:
        spline_points.append((free_path[y_index], y[y_index]))
# Sort the spline points so that Y is increasing (it will crash otherwise)
spline_points_x, spline_points_y = zip(*spline_points)
new_indices = np.argsort(spline_points_y)
spline_points_x = np.array(spline_points_x)[new_indices]
spline_points_y = np.array(spline_points_y)[new_indices]
# Fit a spline to the points (switching X and Y because Y is increasing, not X)
spline = splrep(spline_points_y, spline_points_x)
# Evaluate the spline on the full Y range
spline_x = splev(y, spline)

# DISPLAY
# Display the relevant edges and paths
plt.plot(left_edge, y, right_edge, y, spline_x, y)
# Scatter plot the obstacles
plt.scatter(*zip(*obstacles))
# Show the combined plot on screen
plt.show()
