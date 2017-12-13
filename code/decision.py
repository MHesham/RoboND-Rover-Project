import numpy as np
import math
import time

# The velocity and angular velocity sampling period in seconds.
SAMPLE_PERIOD_SECONDS = 3.0

# A velocity threshold used to detect a forward motion stuck state.
STUCK_VEL_THRESHOLD = 0.1

# An angular velocity threshold used to detect an inplace turn motion stuck
# state.
STUCK_ANGULAR_VEL_THRESHOLD = 0.25

# The velocity threshold of deciding whether the rover is nearly stopped.
# A value of velocity less than this threshold is considered a stopped state.
BRAKE_VEL_THRESHOLD = 0.2

# Rover minimum steering angle in degrees. Negative value is left turn.
ROVER_STEER_MIN = -15
# Rover maximum steering angle in degrees. Positive value is right turn.
ROVER_STEER_MAX = 15


def detect_stuck(Rover):

    capture_sample = False

    if Rover.prev_sample_time is None:
        capture_sample = True
    elif time.time() - Rover.prev_sample_time > SAMPLE_PERIOD_SECONDS:
        capture_sample = True
        Rover.is_vel_stuck = False
        Rover.is_steer_stuck = False
        if (Rover.mode == 'forward'):
            if (Rover.throttle > 0) and (math.fabs(Rover.vel) < STUCK_VEL_THRESHOLD):
                Rover.is_vel_stuck = True
        angular_vel = (Rover.yaw - Rover.prev_yaw) / SAMPLE_PERIOD_SECONDS
        if (Rover.mode == 'stop'):
            if (Rover.steer != 0) and (math.fabs(angular_vel) < STUCK_ANGULAR_VEL_THRESHOLD):
                Rover.is_steer_stuck = True

    if capture_sample:
        Rover.prev_sample_time = time.time()
        Rover.prev_vel = Rover.vel
        Rover.prev_yaw = Rover.yaw

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function


def has_good_vision(Rover):
    return len(Rover.nav_angles) >= Rover.go_forward


def has_poor_vision(Rover):
    return len(Rover.nav_angles) <= Rover.stop_forward


def rover_break(Rover):
    Rover.mode = 'stop'
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.steer = 0


def rover_go_forward(Rover):
    Rover.mode = 'forward'
    # If mode is forward, navigable terrain looks good
    # and velocity is below max, then throttle
    if Rover.vel < Rover.max_vel:
        # Set throttle value to throttle setting
        Rover.throttle = Rover.throttle_set
    else:  # Else coast
        Rover.throttle = 0
    Rover.brake = 0
    # Set steering to average angle clipped to the range +/- 15
    Rover.steer = np.clip(
        np.mean(Rover.nav_angles * 180 / np.pi), ROVER_STEER_MIN, ROVER_STEER_MAX)


def rover_inplace_turn(Rover, Degree):
    Rover.throttle = 0
    # Release the brake to allow turning
    Rover.brake = 0
    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
    Rover.steer = Degree  # Could be more clever here about which way to turn


def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    detect_stuck(Rover)

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check stuck state and the extent of navigable terrain
            if Rover.is_vel_stuck or has_poor_vision(Rover):
                rover_break(Rover)
            else:
                rover_go_forward(Rover)
        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            if Rover.vel <= BRAKE_VEL_THRESHOLD:
                # Now we're stopped and we have vision data to see if there's a path forward
                # If we're stopped but see sufficient navigable terrain in front then go!
                if not Rover.is_vel_stuck and has_good_vision(Rover):
                    rover_go_forward(Rover)
                else:
                    rover_inplace_turn(Rover, ROVER_STEER_MIN)
    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True

    return Rover
