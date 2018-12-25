#!/usr/bin/env python3
"""Generate a spline to avoid a set of obstacles in the 2D plane"""

from autograd import grad
import autograd.numpy as np
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize, basinhopping
import zmq


def path_loss(path, x, free):
    """Given a set of Y values corresponding to the predefined X range, return a loss that optimizes distance from obstacles as well as the ideal path"""
    # Get the mean squared error of the path from the constant free path
    path_mean_squared_error = np.mean(np.square(free - path))
    # Create a list containing things to add up to produce the final loss
    loss_values = [path_mean_squared_error]
    # Iterate over the path, calculating differences from one point to the next so the derivative is incorporated into the loss
    for point_index in range(len(path) - 1):
        # Add the squared derivative to the loss list
        loss_values.append(np.square((path[point_index] - path[point_index + 1]) / (x[point_index] - x[point_index + 1])) * 10 / (len(path) - 1))
    # Iterate over the obstacles, adding to the loss
    for obstacle_x, obstacle_y in obstacles:
        # Get the Pythagorean distance from each point on the path to this obstacle
        distances = np.sqrt(np.square(obstacle_x - x) + np.square(obstacle_y - path))
        # Invert all of these distances so closer is worse, multiply them by a constant, and add them to the loss
        loss_values.append(np.mean(1 / distances) * 800)
    # Return the aggregated loss
    return np.sum(loss_values)


# Connect to the simulation using ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5556')
# Infinite loop during which we receive packets from the Unity simulation
while True:
    # # Get a message from the simulation
    # message = socket.recv_json()
    # # Take the obstacles from the message
    # obstacles = [(vector['x'], vector['y']) for vector in message['obstacles']]
    obstacles = [(14.517830848693848, 5.28299617767334), (20.92901611328125, -5.1113810539245605), (24.710485458374023, -0.36252260208129883),
                 (30.900638580322266, 8.074536323547363), (23.905414581298828, -3.4248924255371094), (39.547122955322266, -0.677603542804718)]
    # Get the X and Y positions of the obstacles individually
    obstacles_x, obstacles_y = [np.array(value_list) for value_list in zip(*obstacles)]
    # Create a range of fairly widely spaced X axis values to optimize
    x = np.arange(0, 30, 1)
    # # The optimal free path (assuming no obstacles) should be a line to the provided endpoint
    # end_point_x = message['endPoint']['x']
    # end_point_y = message['endPoint']['y']
    end_point_x = 40
    end_point_y = -2
    free_path_line = np.polyfit([0, end_point_x], [0, end_point_y], deg=1)
    print(free_path_line)
    free_path = np.poly1d(free_path_line)(x)
    # Minimize the loss to produce an optimal path
    path = minimize(path_loss, x0=free_path, args=(x, free_path), method='TNC', jac=grad(path_loss)).x
    # Fit a spline to the points
    spline = splrep(x, path)
    # Evaluate the spline on a denser X range
    dense_x = np.arange(0, 30, 0.05)
    spline_y = splev(dense_x, spline)
    import matplotlib.pyplot as plt
    plt.scatter(obstacles_x, obstacles_y)
    plt.plot(x, path)
    plt.show()
    # # Send the resulting list of points, formatted in a JSON wrapper
    # wrapper = {'pathPoints': [{'x': point[0], 'y': point[1]} for point in zip(dense_x, spline_y)]}
    # socket.send_json(wrapper)
