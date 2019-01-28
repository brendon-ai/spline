#!/usr/bin/env python3
"""Generate a spline to avoid a set of obstacles in the 2D plane"""

from autograd import grad
import autograd.numpy as np
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize, basinhopping
import zmq


def path_loss(path, x, free, points):
    """Given a set of Y values corresponding to the predefined X range, return a loss that optimizes distance from obstacles as well as the ideal path"""
    # Extract the points from the list
    right0, left0, right1, left1 = points
    # Get the mean squared error of the path from the constant free path
    path_mean_squared_error = np.mean(np.square(free - path))
    # Create a list containing things to add up to produce the final loss
    loss_values = [path_mean_squared_error]
    # On the first point, IT MUST BE ZERO (that's where the bike is)
    loss_values.append(100_000 * np.square(path[0]))
    # Iterate over the path, calculating differences from one point to the next so the derivative is incorporated into the loss
    for point_index in range(len(path) - 1):
        # Add the squared derivative to the loss list (multiplied by a constant and divided by the number of derivatives)
        loss_values.append(np.square((path[point_index] - path[point_index + 1]) / (x[point_index] - x[point_index + 1])) * 10 / (len(path) - 1))
    # Iterate over the obstacles, adding to the loss
    for obstacle_x, obstacle_y in obstacles:
        # Get the Pythagorean distance from each point on the path to this obstacle
        distances = np.sqrt(np.square(obstacle_x - x) + np.square(obstacle_y - path))
        # Invert all of these distances so closer is worse, multiply them by a constant, and add them to the loss
        loss_values.append(np.mean((4 / distances) ** 2) * 20)
    # Return the aggregated loss
    return np.sum(loss_values)


# Connect to the simulation using ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5556')
# Infinite loop during which we receive packets from the Unity simulation
while True:
    # Get a message from the simulation
    message = socket.recv_json()
    # Take the obstacles from the message
    obstacles = [(vector['x'], vector['y']) for vector in message['obstacles']]
    if message['yCommand'] != 0:
        print(message['yCommand'])
    # Example output
    wrapper = {'frontAngle': 0.1, 'backAngle': 0.1, 'frontSpeed': 300, 'backSpeed': 300}
    socket.send_json(wrapper)
    # # Get the X and Y positions of the obstacles individually
    # obstacles_x, obstacles_y = [np.array(value_list) for value_list in zip(*obstacles)]
    # # Create a range of fairly widely spaced X axis values to optimize
    # x = np.arange(0, 40, 2)
    # # Get the four points marking out the road edges
    # right0, left0, right1, left1 = [np.array([message[point_name]['x'], message[point_name]['y']]) for point_name in ['right0', 'left0', 'right1', 'left1']]
    # # The optimal free path (assuming no obstacles) should be the average of the two road edges
    # average0 = (right0 + left0) / 2
    # average1 = (right1 + left1) / 2
    # free_path_line = np.polyfit([average0[0], average1[0]], [average0[1], average1[1]], deg=1)
    # free_path = np.poly1d(free_path_line)(x)
    # # Try a few distortions of the main free path, and choose the one that produces the best loss
    # paths = []
    # for possible_free_path_line in [free_path_line, [free_path_line[0] + 0.05, free_path_line[1]], [free_path_line[0] - 0.05, free_path_line[1]]]:
    #     # Convert the line to an actual path
    #     possible_free_path = np.poly1d(possible_free_path_line)(x)
    #     # Minimize the loss to produce an optimal path
    #     possible_path = minimize(path_loss, x0=possible_free_path, args=(x, free_path, (right0, left0, right1, left1)), method='TNC', jac=grad(path_loss)).x
    #     paths.append(possible_path)
    # # Choose the path with the lowest loss
    # path = min(paths, key=lambda p: path_loss(p, x, free_path, (right0, left0, right1, left1)))
    # # Fit a spline to the points
    # spline = splrep(x, path)
    # # Evaluate the spline on a denser X range
    # dense_x = np.arange(0, 40, 0.05)
    # spline_y = splev(dense_x, spline)
    # # Send the resulting list of points, formatted in a JSON wrapper
    # wrapper = {'pathPoints': [{'x': point[0], 'y': point[1]} for point in zip(dense_x, spline_y)]}
    # socket.send_json(wrapper)
