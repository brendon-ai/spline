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
B = np.array([[0], [-1 / CENTER_OF_MASS_HEIGHT], [1]])
# Loss matrix for states
Q = 1 * np.diag([1 / (0.3 ** 2), 1 / (0.1 ** 2), 1 / (0.5 ** 2)])
# Loss matrix for inputs
R = np.array([[1 / (5 ** 2)]])
# Calculate LQR optimal control policy
K, _, _ = control.lqr(A, B, Q, R)


def calculate_wheel_velocity_vectors(forward_speed, orthogonal_speed, heading_speed):
    """Given speed values for X, Y, and heading, as well as the heading, calculate the corresponding speed values along the X and Y axes for each of the two omnidirectional wheels"""
    # Calculate the difference between the orthogonal speeds of the two wheels, according to the heading speed
    relative_orthogonal_speed = WHEELBASE * heading_speed
    orthogonal_speed_front = orthogonal_speed + (0.5 * relative_orthogonal_speed)
    orthogonal_speed_back = orthogonal_speed - (0.5 * relative_orthogonal_speed)
    # There should never be a difference in the forward speeds of the two wheels
    # The wheel spin direction is inverted relative to the axes of the world
    return -forward_speed, -forward_speed, orthogonal_speed_front, orthogonal_speed_back


def cartesian_to_polar_velocity(forward_speed, orthogonal_speed, current_angle):
    """Given a velocity vector for a wheel, calculate the required angle and speed (in radians per second) of the swerve unit"""
    # Calculate the ideal angle and total speed geometrically
    # Make the angle negative because Bullet has anticlockwise angles
    positive_speed_angle = np.arctan2(orthogonal_speed, forward_speed)
    total_speed = np.sqrt(np.square(forward_speed) + np.square(orthogonal_speed)) / WHEEL_RADIUS
    # If it is more efficient to go to the directly opposite angle and run the wheel backwards, do that
    # Make sure to bound the error in the range (-pi, pi)
    error = bound_angle(positive_speed_angle - current_angle)
    # There are special cases for angles closest to pi and -pi
    if error < np.pi * -0.5:
        corrected_error = error + np.pi
        corrected_speed = total_speed * -1
    elif error < np.pi * 0.5:
        corrected_error = error
        corrected_speed = total_speed
    else:
        corrected_error = error - np.pi
        corrected_speed = total_speed * -1
    # If the angle is a long way away from the target, wait to apply force until it is reasonably stable
    if corrected_error > 0.1:
        corrected_speed = 0
    # Add the corrected error to the current angle to get the desired target angle
    corrected_angle = current_angle + corrected_error
    # Return the modified angle and speed
    return corrected_angle, corrected_speed


def control_vehicle(x_speed, y_speed, heading, tilt, tilt_speed, current_angle_front, current_angle_back, x_command, y_command, heading_command):
    """Given commands and state values, calculate the controls to use for the vehicle"""
    # Get the delta time since the last run of this loop
    current_time = time.time()
    global last_time
    delta_time = current_time - last_time
    # Store the current time for next run
    last_time = current_time
    # Rotate the command direction vector so it is aligned with the vehicle's heading
    forward_command, orthogonal_command = rotate_vector(x_command, y_command, heading)
    # Do likewise for the current X and Y speeds
    forward_speed, orthogonal_speed = rotate_vector(x_speed, y_speed, heading)
    # Get the state vector to run the balancing state space controller
    state_vector = np.array([tilt, tilt_speed, orthogonal_speed])
    # Get the reference vector, which should encourage 0 tilt with 0 change, and the desired orthogonal speed
    reference_vector = np.array([0, 0, orthogonal_command])
    # Run the state space controller to get the desired orthogonal acceleration
    orthogonal_acceleration = np.matmul(K, reference_vector - state_vector).tolist()[0]
    # Get the desired speed by adding the acceleration multiplied by delta time to the current speed
    global idealized_orthogonal_speed
    idealized_orthogonal_speed += (orthogonal_acceleration * delta_time)
    print(orthogonal_command, orthogonal_speed, tilt, tilt_speed)
    # Calculate the wheel velocities needed to satisfy the direction commands
    forward_speed_front, forward_speed_back, orthogonal_speed_front, orthogonal_speed_back = calculate_wheel_velocity_vectors(forward_command, idealized_orthogonal_speed, heading_command)
    # Calculate the wheel directions and speeds needed to create these velocities
    angle_front, total_speed_front = cartesian_to_polar_velocity(forward_speed_front, orthogonal_speed_front, current_angle_front)
    angle_back, total_speed_back = cartesian_to_polar_velocity(forward_speed_back, orthogonal_speed_back, current_angle_back)
    # Return the chosen angles and speeds
    return angle_front, angle_back, total_speed_front, total_speed_back


def bound_angle(angle):
    """Bound an angle in the range (-pi, pi)"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def rotate_vector(x, y, angle):
    """Rotate a vector by a specified angle"""
    return (x * np.cos(angle)) + (y * np.sin(angle)), (x * np.sin(angle) * -1) + (y * np.cos(angle))
