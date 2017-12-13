## Project: Search and Sample Return

**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[threshed_rock1]: ./output/threshed_rock1.jpg
[threshed_rock2]: ./output/threshed_rock2.jpg
[threshed_rock2_default_val]: ./output/threshed_rock2_default_val.jpg
[enhanced_stats]: ./output/enhanced_stats.jpg
[hitting_rock]: ./output/hitting_rock.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
I had first to work through the quizes to understand and light up the perception pipeline step by step:
- Define `color_thresh` to detect navigable terrain.
- Define `prespect_transform` source and destination.
- Define `rover_cords` to map the captured image from image space to rover-centric (camera) space.
- Define `rotate_pix` and `translate_pix` to transform pixels from rover space camera space to the world space.

Following is how I addressed each Rubric team.

#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

1. Recorded 2 datasets with different resolution
  - Recorded FullHD quality dataset named 'test_dataset2' with resolution 1920x1080 with Fantastic image quality. Output images size ranged from 2.5KB to 8.2KB.
  - Recorded QHD quality dataset named 'test_dataset3' with resolution 2560x1440 with Fantastic image quality. Output images range from 5.1KB to 8.7KB.
  - Noticed that the output image becomes sharpest with QHD as expected.
  - Noticed also that the darker the scene, the smaller the image. I believe that relates to how JPEG compresses images.
2. Modified the notebook to define a global dataset path to use across the experiment, experimented with 3 dataset.
```py
sample_dataset_name = 'test_dataset'
recorded_hd_dataset_name = 'test_dataset2'
recorded_qhd_dataset_name = 'test_dataset3'
dataset_name = recorded_hd_dataset_name
dataset_dir = '../' + dataset_name
```
3. Defined `perspect_transform` source and destination as follows:
```py
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
```
4. Applied perspective transform as follows 
```py
  warped = perspect_transform(img, source, destination)
```
5. Defined a new color thresholding function `color_threshold2` to allow imposing an upper and lower threshold, pixels that lie between the limits are set to 1, other pixels are set to 0.
```py
def color_thresh2(img, rgb_upper_thresh=(255, 255, 255), rgb_lower_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    in_thresh = (img[:,:,0] > rgb_lower_thresh[0]) \
                & (img[:,:,1] > rgb_lower_thresh[1]) \
                & (img[:,:,2] > rgb_lower_thresh[2]) \
                & (img[:,:,0] < rgb_upper_thresh[0]) \
                & (img[:,:,1] < rgb_upper_thresh[1]) \
                & (img[:,:,2] < rgb_upper_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[in_thresh] = 1
    # Return the binary image
    return color_select
```
6. Examine the color ranges of a rock Using the interactive matplotlib by means of `%matplotlib notebook`. Observed an RGB color range between (130,100,0) and (220,180,80).
7. Defined the navigable terrain, obstacle and rock thresholding as follows:
```py
    nav_img = color_thresh(warped, (160,160,160))
    obst_img = 1 - nav_img
    rock_img = color_thresh2(warped, (220,180,80), (130,100,0))
```
8. Experimented with 2 different rock containing images, below is the output that shows the thresholding result for navigable terrain, obstacles and rocks:
![Thresholded Warped Image Example 2 with Default][threshed_rock2_default_val]
9. Experimented with different threshold for navigable terrain than the provided (160,160,160), could arrive at a bit better result with (130,100,80). Following is the result with that new threshold:
![Thresholded Warped Image Example 2][threshed_rock2]
The previous figure shows that I was able to correctly identify a larger area of navigable terrain, but very tiny part of the rock was identified as navigable terrain as well, not as good as expected but close.
![Thresholded Warped Image Example 1][threshed_rock1]

Here is the code required to draw the above figures:
```py
NAV_COLOR_THRESHOLD = (130,100,80)
#NAV_COLOR_THRESHOLD = (160,160,160)
nav_thresh = color_thresh(warped,NAV_COLOR_THRESHOLD)
obst_thresh = 1 - nav_thresh
rock_thresh = color_thresh2(warped, (220,180,80), (130,100,0))
vision_nav_thresh = color_thresh(image,NAV_COLOR_THRESHOLD)
vision_obst_thresh = 1 - vision_nav_thresh
vision_rock_thresh = color_thresh2(warped, (220,180,80), (130,100,0))
vision_image = np.dstack((vision_obst_thresh, vision_rock_thresh, vision_nav_thresh)).astype(np.float)
fig = plt.figure(figsize=(9.5,9))
plt.subplot(321)
plt.imshow(image)
plt.title('Camera', fontsize=15)
plt.subplot(322)
plt.imshow(vision_image)
plt.title('Vision', fontsize=15)
plt.subplot(323)
plt.title('Warped', fontsize=15)
plt.imshow(warped)
plt.subplot(324)
plt.title('Navigable Terrain Threshold', fontsize=15)
plt.imshow(nav_thresh, cmap='gray')
plt.subplot(325)
plt.title('Obstacles Threshold', fontsize=15)
plt.imshow(obst_thresh, cmap='gray')
plt.subplot(326)
plt.title('Rocks Threshold', fontsize=15)
plt.imshow(rock_thresh, cmap='gray')
fig.savefig('../output/threshed.jpg')
```
#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

Filled in the required code in `process_image` under the `TODO` comment to implement the perception pipeline.

```py
    # TODO: 
    # 1) Define source and destination points for perspective transform
```
Source an destination where defined earlier in the notebook and they were referenced in `process_image`.

```py
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
```

```py
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_img = color_thresh(warped, NAV_COLOR_THRESHOLD)
    obst_img = 1 - nav_img
    rock_img = color_thresh2(warped, (220,180,80), (130,100,0))
```

```py
    # 4) Convert thresholded image pixel values to rover-centric coords
    nav_x, nav_y = rover_coords(nav_img)
    obst_x, obst_y = rover_coords(obst_img)
    rock_x, rock_y = rover_coords(rock_img)
```

```py
    # 5) Convert rover-centric pixel values to world coords
    scale = dst_size * 2
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]
    worldsize = data.worldmap.shape[0]
    nav_x_world, nav_y_world = pix_to_world(nav_x, nav_y, xpos, ypos, yaw, worldsize, scale)
    obst_x_world, obst_y_world = pix_to_world(obst_x, obst_y, xpos, ypos, yaw, worldsize, scale)
    rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, worldsize, scale)
```

```py
    # 6) Update worldmap (to be displayed on right side of screen)
    data.worldmap[obst_y_world, obst_x_world, 0] += 1
    data.worldmap[rock_y_world, rock_x_world, 1] += 1
    data.worldmap[nav_y_world, nav_x_world, 2] += 1    
```
Here we mark on the worldmap RGB image the type of each type by using the following scheme for RGB channels: R:obstacle, G:rock, B:navigable. The 3 color intensities get blended on display. The perfect scenario is when every pixel gets identified correctly every time, which doesn't happen in real. What happens is that when error happens and a certain pixel get mis-categorized, a pixel will have a blend of the 3 channels as the final output. e.g. certain pixel can get identified as rock and sometimes as obstacle at the same time due to the intersection of 2 ranges.

#### Training video output environment:
- Dataset Recording Resolution: 1920x1080
- Dataset Recording Quality: Fantastic

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

#### 1.1. Perception Step

Following is the `perception_step()` implementation in `perception.py`. The input is the current rover state via the parameter `Rover`. The output is the updated rover state where the update happens in-place inside the passed in `Rover`. The function `perception_step()` takes the current camera image as captured in `Rover` and process it in the same pipeline used in the notebook analysis. The steps and the code below are very similar to that in the notebook analysis with the follow major differences and extras:
1. `Rover.vision_image` gets updated with the thresholded binary image of the navigable terrain, obstacles and rocks.
2. All pixels get transformed from the rover space to the world space by means of `pix_to_world()` function.
3. Compute the polar coordinates of each navigable pixel and update `Rover` `Rover.nav_dists` and `Rover.nav_angles` accordingly. Those distances and angles are in the rover space.
4. Optimization Map Fidelity: To enhance fidelity a little bit, a helper function `is_valid_perception_state()` was implemented to aid in identifying whether the current camera image is good enough to be mapped or not. The constants `PITCH_DRIFT_THRESHOLD` and `ROLL_DRIFT_THRESHOLD` define the rover pitch and roll threshold after which the camera image is considered not valid for mapping.

```py
PITCH_DRIFT_THRESHOLD = 1.0
ROLL_DRIFT_THRESHOLD = 0.25

def is_valid_perception_state(Rover):
    if Rover.pitch > PITCH_DRIFT_THRESHOLD and Rover.pitch < (360.0 - PITCH_DRIFT_THRESHOLD):
        return False
    if Rover.roll > ROLL_DRIFT_THRESHOLD and Rover.roll < (360.0 - ROLL_DRIFT_THRESHOLD):
        return False
    return True
```

Following is `perception_step()` implementation in which we update `Rover.worldmap` only if `is_valid_perception_state()` returns `True`.
```py
# Apply the above functions in succession and update the Rover state accordingly


def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
    # 1) Define source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size,
                               image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] -
                               2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] -
                               2 * dst_size - bottom_offset],
                              ])

    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_thresh = color_thresh(warped, (130, 100, 80))
    obst_thresh = 1 - nav_thresh
    rock_thresh = color_thresh2(warped, (220, 180, 80), (130, 100, 0))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = obst_thresh * 255
    Rover.vision_image[:, :, 1] = rock_thresh * 255
    Rover.vision_image[:, :, 2] = nav_thresh * 255

    # 5) Convert map image pixel values to rover-centric coords
    obst_x, obst_y = rover_coords(obst_thresh)
    rock_x, rock_y = rover_coords(rock_thresh)
    nav_x, nav_y = rover_coords(nav_thresh)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = dst_size * 2
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    worldsize = Rover.worldmap.shape[0]
    nav_x_world, nav_y_world = pix_to_world(
        nav_x, nav_y, xpos, ypos, yaw, worldsize, scale)
    obst_x_world, obst_y_world = pix_to_world(
        obst_x, obst_y, xpos, ypos, yaw, worldsize, scale)
    rock_x_world, rock_y_world = pix_to_world(
        rock_x, rock_y, xpos, ypos, yaw, worldsize, scale)

    # Optimizing Map Fidelity Tip:
    # Your perspective transform is technically only valid when roll and pitch
    # angles are near zero. If you're slamming on the brakes or turning hard
    # they can depart significantly from zero, and your transformed image will
    # no longer be a valid map. Think about setting thresholds near zero in roll
    # and pitch to determine which transformed images are valid for mapping.
    if is_valid_perception_state(Rover):
        # 7) Update Rover worldmap (to be displayed on right side of screen)
        Rover.worldmap[obst_y_world, obst_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[nav_y_world, nav_x_world, 2] += 1
    else:
        print('Invalid perception state pitch={} roll={}'.format(
            Rover.pitch, Rover.roll))

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    dist, angles = to_polar_coords(nav_x, nav_y)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    return Rover
```

#### 1.2. Decision Step
1. Implement helper function `detect_stuck()` to aid in detecting conditions in which the rover is stuck in either moving forward or in-place turning. The idea is to sample both velocity and angular velocity over a sampling period `SAMPLE_PERIOD_SECONDS` and if either are found to be below the thresholds `STUCK_VEL_THRESHOLD` or `STUCK_ANGULAR_VEL_THRESHOLD` then we know that we are stuck, and update the `Rover.is_vel_stuck` or `Rover.is_steer_stuck` accordingly before returning. The `decision_step()` function should use these 2 flags to aid in its decision making. Following is `detect_stuck()` implementation:

```py

# The velocity and angular velocity sampling period in seconds.
SAMPLE_PERIOD_SECONDS = 3.0

# A velocity threshold used to detect a forward motion stuck state.
STUCK_VEL_THRESHOLD = 0.1

# An angular velocity threshold used to detect an inplace turn motion stuck
# state.
STUCK_ANGULAR_VEL_THRESHOLD = 0.25


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
```
2. Following is 2 helper functions `has_good_vision()` and `rover_inplace_turn()` which are used by `decision_step()` to help deciding whether the rover has good enough vision to go forward or has poor vision and it has to stop and rethink.
```py
def has_good_vision(Rover):
    return len(Rover.nav_angles) >= Rover.go_forward


def has_poor_vision(Rover):
    return len(Rover.nav_angles) <= Rover.stop_forward
```
3. The decision logic provided as part of the project in `decision_step()` was refactored into another 3 helper functions `rover_break()`, `rover_go_forward()` and `rover_inplace_turn()`. I mainly did that refactoring to simplify the logic code and make it more readable.
```py
# Rover minimum steering angle in degrees. Negative value is left turn.
ROVER_STEER_MIN = -15

# Rover maximum steering angle in degrees. Positive value is right turn.
ROVER_STEER_MAX = 15

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
```
4. Follwing is the `decision_step()` implementation. It had a couple of changes around the stuck condition detection and resolve:
  - Call detect_stuck(Rover) to detect the velocity or steer stuck conditions.
  - Transition from `forward` to `stop` state on `Rover.is_vel_stuck`.
  - Stay in the `stop` state as long as the rover doesn't have good vision or still in `Rover.is_vel_stuck` is still true. Note: Currently, the `Rover.is_steer_stuck` flag is not used, it was initially used for experimentation. It was left there for leverage in the future.
```py
# The velocity threshold of deciding whether the rover is nearly stopped.
# A value of velocity less than this threshold is considered a stopped state.
BRAKE_VEL_THRESHOLD = 0.2

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

```

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

The autonomous driving run as recorded in the output video has the following:
- Environment:
  - OS: Ubuntu 17.1 (Not VM)
  - PC: Intel i5 6600K QuadCore, Nvidia 1060 GTX, 16GB RAM.
- Settings:
  - Resolution: 1920 x 1080
  - Quality: Fantastic
- Results:
  - Mapped 54% of the world map with 60% fidelity in 109 seconds.
  - Located 1 rock only, with no picking.

HUD Enhancements:
- Writing more statistics on the worldmap overlay to aid in debugging the rover behavior by showing the following: Rover mode/state, number of navigable pixels, velocity and steer stuck flags, current steer angle.
- Draw the Rover location on the worldmap as a yellow dot.

![Worldmap Overlay Enhancements][enhanced_stats]

Optimizations and Results:
- Optimizing Map Fidelity:
  - Bad image detection:
    - Implemented a logic in `perception_step()` to detect if the rover wasn't near 0 pitch and roll, such images were not used for mapping to the `Rover.worldmap`.
    - Results: Noticed slight improvement in fidelity overall, the more my pitch and roll thresholds are close to 0 the better fidelity I am getting. I couldn't go to zero, because the rover will have a tiny pitch due to forward acceleration.
  - Image thresholding improvement:
    - Expiremented with different navigable terrain color threshold other than the suggested (160,160,160). Found (130, 100, 80) to perform better in some scenarios but not always.
    - Reults: There was no noticable improvement in mapping fidelity.
- Optimizing Time:
  - Increased the rover max velocity to 4M/s at the beginning, but found that to be too fast for the rover decision tree.
  - Increased the `Rover.stop_forward` and `Rover.go_forward` from to 50 and 500 to 1000 and 2500 to account for the increase in speed and to give the rover enough time to respond.
  - Problem with 4M/s speed is that it was too fast sometimes for the rover to make turns to avoid rocks and that lead to getting stuck alot, which was the motivation behind implementing the stuck detection and resolve logic.
  - Results: At the end I settled down on 3M/s max velocity, found it to be more stable and effective given the current implemented logic.
- Optimizing for Finding All Rocks:
  - Make the rover biased to a specific side of the wall when deciding the steer angle by limiting the steer angle to the range [-15,10] which is supposed to bias the rover to left turns.
  ```py
      Rover.steer = np.clip(
        np.mean(Rover.nav_angles * 180 / np.pi), ROVER_STEER_MIN, ROVER_STEER_MAX)
  ```
  - Results: That resulted in map coverage % sometimes, but at other times it increased the chance of the rover getting stuck or hitting the wall because it couldn't steer right enough.

There are a couple of scenarios in which the rover will very likely to go wrong:
- Keep moving around in circles and missing sharp exits:
  - It was observed that at some wide open places in the worldmap, if the rover starts from a specific corner, it will keep choosing to do left turn and it will ignore a sharp right exit. The only way for the rover to detect that exit is if it coming from a different direction in which case the camera will see that exit terrain and be considered in the decision making. 
- Inability to evade fast approaching rocks:
  - Hitting a rock that is directly infront of the rover when going forward. The image below illustrates that the rover recognized alot of navigable terrain (12K pixel), but the rock is dividing those pixels in half. Using mean angle to steer will likely fail, and the rover will drive into the rock and get stuck. Luckly the rover will be able to detect that stuck state and resolve it as illustrated earlier.
![Hitting Rock Scenario][hitting_rock]



