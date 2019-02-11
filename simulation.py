#!/usr/bin/env python3
"""Simulate an environment for testing of control algorithms"""

import time

import numpy as np
import pybullet_data
import pybullet as p

from controller import control_vehicle

# Constants for the IDs of various joints
FRONT_SWERVE, FRONT_DRIVE, BACK_SWERVE, BACK_DRIVE = range(4)
# Time step length
DELTA_TIME = 1 / 240

# Create a client and connect to the GUI
client = p.connect(p.GUI)
# Enable loading of models included by default (the plane)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Configure the basic parameters and load the plane
p.setGravity(0, 0, -9.81)
plane_id = p.loadURDF('plane.urdf')
# Load the bike and drop it into the scene
start_position = [0, 0, 0.475]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
bike_id = p.loadURDF("bike.xml", start_position, start_orientation)
# Remember the last tilt so we can calculate the derivative
last_tilt = 0
# Run the simulation for a defined number of steps
for i in range(100_000):
    # Get the current state of the bike
    bike_position, bike_orientation_quaternion = p.getBasePositionAndOrientation(bike_id)
    # Specifically, we want the X and Y positions
    x_pos, y_pos, _ = bike_position
    # Convert the orientation to Euler angles, and get the tilt angle and heading
    tilt, _, heading = p.getEulerFromQuaternion(bike_orientation_quaternion)
    # Get the current velocity of the bike (positional only)
    bike_position_speed, _ = p.getBaseVelocity(bike_id)
    # We want the X and Y speeds (but for some reason they are inverted)
    x_speed, y_speed, _ = bike_position_speed
    # For some reason the Y speed is inverted
    y_speed *= 1
    # Calculate the tilt speed numerically
    tilt_speed = (tilt - last_tilt) / DELTA_TIME
    # Update the previous tilt value
    last_tilt = tilt
    # Get the current angles of the swerve joints
    current_angle_front, _, _, _ = p.getJointState(bike_id, FRONT_SWERVE)
    current_angle_back, _, _, _ = p.getJointState(bike_id, BACK_SWERVE)
    # Get the desired control values according to the current states
    angle_front, angle_back, total_speed_front, total_speed_back = control_vehicle(x_speed, y_speed, heading, tilt, tilt_speed, current_angle_front, current_angle_back, 1, 0, 0.3)
    # Set the desired angles of the swerve motors
    p.setJointMotorControlArray(bodyUniqueId=bike_id, jointIndices=[FRONT_SWERVE, BACK_SWERVE], controlMode=p.POSITION_CONTROL, targetPositions=[angle_front, angle_back], forces=[20] * 2)
    # Set the desired speeds of the drive motors
    p.setJointMotorControlArray(bodyUniqueId=bike_id, jointIndices=[FRONT_DRIVE, BACK_DRIVE], controlMode=p.VELOCITY_CONTROL, targetVelocities=[total_speed_front, total_speed_back], forces=[10] * 2)
    # Step the physics simulation
    p.stepSimulation()
    # Wait a small amount of time before moving on to the next step
    time.sleep(DELTA_TIME)
# Disconnect from the server at the very end
p.disconnect()
