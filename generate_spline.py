#!/usr/bin/env python3
"""Generate a spline to avoid a set of obstacles in the 2D plane"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev

# DEFINITIONS
# Create a range of Y axis values to plot on
y = np.arange(-10, 10, 0.01)
# Define the edges of the road in terms of linear functions of the form x=my+b (all units are in meters)
left_edge = 0.02 * y ** 2 + 0.5 * y + 2
right_edge = 0.02 * y ** 2 + 0.5 * y
# Define a set of obstacles present within the road
obstacles = [(-1.5, -5), (1, -1), (4, 5)]

# COMPUTATIONS
# The optimal free path (assuming no obstacles) should be the average of the left and right edges
free_path = (left_edge + right_edge) / 2
# Create a list of spline points to add to the winding path, starting with the current position
# Add the current position to the spline
spline_points = [(free_path[0], y[0])]
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
    # Choose the edge point on whichever edge is farther away, and average that and the obstacle point to get a spline point
    if left_closest > right_closest:
        spline_points.append(((left_edge[left_closest_index] + obstacle_x) / 2, (y[left_closest_index] + obstacle_y) / 2))
    else:
        spline_points.append(((right_edge[right_closest_index] + obstacle_x) / 2, (y[right_closest_index] + obstacle_y) / 2))

# Fit a spline to the points (switching X and Y because Y is increasing, not X)
spline_points_x, spline_points_y = zip(*spline_points)
spline = splrep(spline_points_y, spline_points_x)
# Evaluate the spline on the full Y range
spline_x = splev(y, spline)

# DISPLAY
# Display all edges and paths in question
plt.plot(left_edge, y, right_edge, y, free_path, y, spline_x, y)
# Scatter plot the obstacles and the spline points
plt.scatter(*zip(*obstacles))
plt.scatter(*zip(*spline_points))
# Show the combined plot on screen
plt.show()
