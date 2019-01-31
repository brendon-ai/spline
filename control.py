#!/usr/bin/env python3
"""Control the simulated vehicle's stability and movement"""

import numpy as np
import zmq


def calculate_wheel_velocity_vectors(x_speed, y_speed, heading_speed, heading, wheelbase):
    """Given speed values for X, Y, and heading, as well as the heading and wheelbase, calculate the corresponding speed values along the X and Y axes for each of the two omnidirectional wheels"""
    relative_x_speed = wheelbase * heading_speed * np.cos(heading)
    x_speed_front = x_speed + (0.5 * relative_x_speed)
    x_speed_back = x_speed - (0.5 * relative_x_speed)
    relative_y_speed = -1 * wheelbase * heading_speed * np.sin(heading)
    y_speed_front = y_speed + (0.5 * relative_y_speed)
    y_speed_back = y_speed - (0.5 * relative_y_speed)
    return x_speed_front, x_speed_back, y_speed_front, y_speed_back


def cartesian_to_polar_velocity(x_speed, y_speed):
    """Given a velocity vector for a wheel, calculate the required angle and speed of the swerve unit"""
    angle = np.arctan2(x_speed, y_speed)
    total_speed = np.sqrt(np.square(x_speed) + np.square(y_speed))
    return angle, total_speed


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
    # Calculate the wheel velocities needed to satisfy the direction commands
    x_speed_front, x_speed_back, y_speed_front, y_speed_back = calculate_wheel_velocity_vectors(
        message['xCommand'] * 100, message['yCommand'] * 500, message['headingCommand'], 0, 1)
    # Calculate the wheel directions and speeds needed to create these velocities
    angle_front, total_speed_front = cartesian_to_polar_velocity(x_speed_front, y_speed_front)
    angle_back, total_speed_back = cartesian_to_polar_velocity(x_speed_back, y_speed_back)
    print(angle_front, total_speed_front, angle_back, total_speed_back)
    # Send the chosen angles and speeds to the simulation
    wrapper = {'frontAngle': angle_front, 'backAngle': angle_back, 'frontSpeed': total_speed_front, 'backSpeed': total_speed_back}
    socket.send_json(wrapper)
