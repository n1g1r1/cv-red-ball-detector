import cv2 as cv
import numpy as np

# Ball constants
ball_radius = 53.6
ball_distance = 1

# HSV thresholds
hsv_min = np.array([0, 90, 100])
hsv_max = np.array([10, 256, 256])
hsv_min2 = np.array([230, 90, 100])
hsv_max2 = np.array([256, 256, 256])

## def red_blob_detector():

# Some values
active = True

# Get camera image
cameraCapture = cv.VideoCapture(0)
success, frame = cameraCapture.read()

# Save image size for camera coordinates calculation

while success and cv.waitKey(1) == -1 and active:

    # Get the actual image
    success, frame = cameraCapture.read()

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
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15))

    # TODO: Remove noise by Open
    cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)

    # TODO: Close gaps by Close
    cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

    # TODO: Blur the image
    thresholded = cv.medianBlur(thresholded, 3)

    # Show the result
    cv.imshow('Enhanced image', thresholded)

    # TODO: Calculate Hough circles
    # circles = cv.HoughCircles(thresholded, cv.HOUGH_GRADIENT,2, 20, param1=50, param2=20, minRadius=20, maxRadius=200)

    # TODO: Draw each circle
    # circles = np.uint16(np.around(circles))

    # for i in circles[0,:]:
    #     cv.circle(frame, i[0], i[1], i[2], (0,255,0),2)

    #     cv.circle(frame, i[0], i[1], 2, (0,0,255),3)


    # Show result
    cv.imshow('Detected Circles', frame)

cv.destroyAllWindows()        