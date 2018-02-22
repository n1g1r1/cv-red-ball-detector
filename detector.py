#######################################################
# Red ball detector script
# @author: Christian Reichel
# Version: 0.1
# -----------------------------------------------------
# This script detects red blobs or balls and draws
# boundary circles.
# 
# Knowm challenges:
# [ ] Stablilise detecting by balancing white value.
# [ ] Build an calibration function on top.
# [ ] Adjust the HSV values bit more for better 
#     tracking.
#######################################################

# IMPORTS.
import numpy as np
import cv2 as cv

# Params: Ball constants.
ball_radius =   53.6
ball_distance = 1

# Params: HSV thresholds.
hsv_min =   np.array([0, 90, 100])
hsv_max =   np.array([10, 256, 256])
hsv_min2 =  np.array([220, 90, 100])
hsv_max2 =  np.array([256, 256, 256])

# Params: Circle detection params
min_dist_between_centers =  100 # Minimal distance between detected centers
upper_threshold_canny =     100 # The upper threshold for Canny edge detector, 0-255
center_threshold =          40 # Threshold for center detection, 0-255
min_radius =                20 # minimum radius of circles
max_radius =                0 # maximum radius

# Get camera image.
camera = cv.VideoCapture(0)
output, camera_image = camera.read()

while output and cv.waitKey(1) == -1:

    # Blur and resize.
    camera_image = cv.medianBlur(camera_image, 5)
    camera_image = cv.resize(camera_image, (0,0), fx = 0.2, fy = 0.2)

    # Convert Colors + combine thresholded images.
    src_hsv = cv.cvtColor(camera_image, cv.COLOR_BGR2HSV)
    thresholded_lower = cv.inRange(src_hsv, hsv_min, hsv_max)
    thresholded_higher = cv.inRange(src_hsv, hsv_min2, hsv_max2)
    thresholded = cv.add(thresholded_lower, thresholded_higher)

    # Make kernel.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))

    # Remove noise by Open.
    cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)

    # Close gaps by Close.
    cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

    # Blur the image.
    thresholded = cv.medianBlur(thresholded, 5)

    # Calculate Hough circles.
    circles = cv.HoughCircles(thresholded, cv.HOUGH_GRADIENT, 1.2, min_dist_between_centers, param1=upper_threshold_canny, param2=center_threshold, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:

        # Round circle values for discrete circle drawings.
        circles = circles[0]
        circles = np.uint16(np.around(circles))

        # Draw each circle.
        for (x, y, r) in circles:

            # Outer circle
            cv.circle(camera_image, (x, y), r, (0, 255, 0), 2)

            # Center point
            cv.circle(camera_image, (x, y), 2, (0, 0, 255), 3)

    # Show result.
    cv.imshow('Detected Circles', camera_image)
    cv.imshow('Thresholded', thresholded)

    # Get next camera image.
    output, camera_image = camera.read()

# End procedures
camera.release()
cv.destroyAllWindows()        