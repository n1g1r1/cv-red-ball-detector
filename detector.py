import cv2 as cv
import numpy as np

# Ball constants
ball_radius =   53.6
ball_distance = 1

# HSV thresholds
hsv_min =   np.array([0, 90, 100])
hsv_max =   np.array([10, 256, 256])
hsv_min2 =  np.array([220, 90, 100])
hsv_max2 =  np.array([256, 256, 256])

# Circle detection params
min_dist_between_centers =  100 # Minimal distance between detected centers
upper_threshold_canny =     100 # The upper threshold for Canny edge detector, 0-255
center_threshold =          40 # Threshold for center detection, 0-255
min_radius =                30 # minimum radius of circles
max_radius =                0 # maximum radius

# Some values
active = True

# Get camera image
cameraCapture = cv.VideoCapture(0)
output, frame = cameraCapture.read()

# Save image size for camera coordinates calculation

while output and cv.waitKey(1) == -1 and active:

    # Get the actual image
    output, frame = cameraCapture.read()

    # TODO: If calibrated: undistort

    # TODO: Blur
    frame = cv.medianBlur(frame, 5)
    
    # TODO: Resize
    frame = cv.resize(frame, (0,0), fx = 0.2, fy = 0.2)

    # Copy image
    src_hsv = frame

    # TODO: Convert Colors + combine thresholded images
    src_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    thresholded_lower = cv.inRange(src_hsv, hsv_min, hsv_max)
    thresholded_higher = cv.inRange(src_hsv, hsv_min2, hsv_max2)
    thresholded = cv.add(thresholded_lower, thresholded_higher)

    # Make kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))

    # TODO: Remove noise by Open
    cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)

    # TODO: Close gaps by Close
    cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

    # TODO: Blur the image
    thresholded = cv.medianBlur(thresholded, 5)

    cv.imshow('Thresholded', thresholded)

    # TODO: Calculate Hough circles
    circles = cv.HoughCircles(thresholded, cv.HOUGH_GRADIENT, 1.2, min_dist_between_centers, param1=upper_threshold_canny, param2=center_threshold, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:

        circles = circles[0]

        # TODO: Draw each circle
        print(circles)
        circles = np.uint16(np.around(circles))

        print(circles)

        for (x, y, r) in circles:
            cv.circle(frame, (x, y), r, (0, 255, 0), 2)

            cv.circle(frame, (x, y), 2, (0, 0, 255), 3)

    # Show result
    cv.imshow('Detected Circles', frame)

cv.destroyAllWindows()        