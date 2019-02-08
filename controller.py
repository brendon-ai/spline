"""Control the simulated vehicle's stability and movement"""

import time

import control
import numpy as np

# Fundamental constants of the vehicle and the environment
WHEELBASE = 1.5
CENTER_OF_MASS_HEIGHT = 0.3
GRAVITY = 9.81
MASS = 16
WHEEL_RADIUS = 0.15

# Store the last time the main loop was run
last_time = time.time()
# Store the idealized orthogonal speed over time
idealized_orthogonal_speed = 0

# Balancing state space control matrices (linearized)
# States: [tilt, tilt speed, orthogonal speed]
# Inputs: [orthogonal acceleration]
# State matrix (defines changing states)
A = np.array([[0, 1, 0], [GRAVITY / CENTER_OF_MASS_HEIGHT, 0, 0], [0, 0, 0]])
# Input matrix (defines modification of states based on inputs)
B = np.array([[0], [-1 / CENTER_OF_MASS_HEIGHT], [1 / MASS]])
# Loss matrix for states
Q = 1 * np.diag([1 / (0.1 ** 2), 1 / (2 ** 2), 1 / (2 ** 2)])
# Loss matrix for inputs
R = np.array([[1 / (40 ** 2)]])
# Calculate LQR optimal control policy
K, _, _ = control.lqr(A, B, Q, R)


def calculate_wheel_velocity_vectors(x_speed, y_speed, heading_speed, heading):
    """Given speed values for X, Y, and heading, as well as the heading, calculate the corresponding speed values along the X and Y axes for each of the two omnidirectional wheels"""
    relative_x_speed = WHEELBASE * heading_speed * np.cos(heading)
    x_speed_front = x_speed + (0.5 * relative_x_speed)
    x_speed_back = x_speed - (0.5 * relative_x_speed)
    relative_y_speed = -1 * WHEELBASE * heading_speed * np.sin(heading)
    y_speed_front = y_speed + (0.5 * relative_y_speed)
    y_speed_back = y_speed - (0.5 * relative_y_speed)
    return x_speed_front, x_speed_back, y_speed_front, y_speed_back


def cartesian_to_polar_velocity(x_speed, y_speed):
    """Given a velocity vector for a wheel, calculate the required angle and speed (in radians per second) of the swerve unit"""
    angle = np.arctan2(x_speed, y_speed)
    total_speed = np.sqrt(np.square(x_speed) + np.square(y_speed)) / WHEEL_RADIUS
    return angle, total_speed


def control_vehicle(x_pos, y_pos, x_speed, y_speed, tilt, tilt_speed, current_angle_front, current_angle_back):
    """Given commands and state values, calculate the controls to use for the vehicle"""
    # Get the delta time since the last run of this loop
    current_time = time.time()
    global last_time
    delta_time = current_time - last_time
    # Store the current time for next run
    last_time = current_time
    # Get the state vector to run the balancing state space controller
    state_vector = np.array([tilt, tilt_speed, y_speed])
    # Get the reference vector, which should encourage 0 tilt with 0 change, and the desired orthogonal speed
    reference_vector = np.array([0, 0, 0])
    # Run the state space controller to get the desired orthogonal acceleration
    orthogonal_acceleration = np.matmul(K, reference_vector - state_vector).tolist()[0]
    # Get the desired speed by adding the acceleration multiplied by delta time to the current speed
    global idealized_orthogonal_speed
    idealized_orthogonal_speed += (orthogonal_acceleration * delta_time)
    # Calculate the wheel velocities needed to satisfy the direction commands
    x_speed_front, x_speed_back, y_speed_front, y_speed_back = calculate_wheel_velocity_vectors(
        idealized_orthogonal_speed, 0, 0, 0)
    # Calculate the wheel directions and speeds needed to create these velocities
    angle_front, total_speed_front = cartesian_to_polar_velocity(x_speed_front, y_speed_front)
    angle_back, total_speed_back = cartesian_to_polar_velocity(x_speed_back, y_speed_back)
    # Return the chosen angles and speeds
    return angle_front, angle_back, total_speed_front, total_speed_back
