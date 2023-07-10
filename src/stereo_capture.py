import cv2
import os
import time

# Set the number of images you want to capture
num_images = 75

# Create directories for left and right images if they don't exist
os.makedirs('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7_1/stereoL', exist_ok=True)
os.makedirs('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7_1/stereoR', exist_ok=True)

# Open the left and right cameras
left_cam = cv2.VideoCapture(1)
right_cam = cv2.VideoCapture(0)

# Set the resolution for each camera
left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Capture images from both cameras
i = 0
while i < num_images:  # limit to 'num_images'

    ret_left, left_frame = left_cam.read()
    ret_right, right_frame = right_cam.read()

    # Save captured images
    cv2.imshow('Left', left_frame)
    cv2.imshow('Right', right_frame)

    if cv2.waitKey(25)==32:
        # left_image_path = os.path.join('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/stereoL/', f'left_{i}.png')
        # right_image_path = os.path.join('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/stereoR/', f'right_{i}.png')

        left_image_path = os.path.join('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7_1/stereoL/', f'left_{i}.png')
        right_image_path = os.path.join('C:/Users/MSDL-DESK-02/Desktop/pyStereo/data/checkboard_10x7_1/stereoR/', f'right_{i}.png')
        cv2.imwrite(left_image_path, left_frame)
        cv2.imwrite(right_image_path, right_frame)
        i += 1
        

# Release the cameras and close all windows
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()