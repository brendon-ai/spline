#!/usr/bin/env python3
"""Generate a spline to avoid a set of obstacles in the 2D plane"""

from autograd import grad
import autograd.numpy as np
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize, basinhopping
import zmq


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


# Connect to the simulation using ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5556')
# Infinite loop during which we receive packets from the Unity simulation
while True:
    # Get a message from the simulation
    message = socket.recv()
    print(message)
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
    # Minimize the loss to produce an optimal path
    path = minimize(path_loss, x0=free_path, method='TNC', jac=grad(path_loss)).x
    # Fit a spline to the points (switching X and Y because Y is increasing, not X)
    spline = splrep(y, path)
    # Evaluate the spline on a denser Y range
    dense_y = np.arange(-10, 2, 0.05)
    spline_x = splev(dense_y, spline)
    # Send the resulting list of points, formatted as a JSON list
    # socket.send_json(list(spline_x))
    socket.send_json('{"pathPoints":[{"x":4.880000114440918,"y":0.09000000357627869},{"x":8.0600004196167,"y":0.09000000357627869},{"x":11.229999542236329,"y":0.7799999713897705},{"x":14.210000038146973,"y":2.4800000190734865},{"x":16.559999465942384,"y":4.670000076293945},{"x":18.229999542236329,"y":6.900000095367432},{"x":19.110000610351564,"y":9.579999923706055},{"x":18.719999313354493,"y":12.329999923706055},{"x":17.280000686645509,"y":14.90999984741211},{"x":15.770000457763672,"y":17.229999542236329},{"x":15.34000015258789,"y":20.040000915527345},{"x":15.800000190734864,"y":22.979999542236329},{"x":17.299999237060548,"y":25.200000762939454},{"x":19.360000610351564,"y":26.969999313354493},{"x":22.329998016357423,"y":27.40999984741211},{"x":25.509998321533204,"y":27.40999984741211},{"x":28.67999839782715,"y":28.100000381469728},{"x":31.65999984741211,"y":29.799999237060548},{"x":34.0099983215332,"y":31.98999786376953},{"x":35.679996490478519,"y":34.220001220703128},{"x":36.560001373291019,"y":36.900001525878909},{"x":36.16999816894531,"y":39.650001525878909},{"x":34.72999954223633,"y":42.22999954223633},{"x":33.21999740600586,"y":44.54999923706055},{"x":32.78999710083008,"y":47.36000061035156},{"x":33.25,"y":50.29999923706055},{"x":34.749996185302737,"y":52.52000045776367},{"x":36.80999755859375,"y":54.290000915527347}]}')

