# This script calculates the transformation from the left camera to the checkerboard frame

import pyzed.sl as sl
import cv2
import numpy as np

# Define the checkerboard dimensions
checkerboard_size = (6, 9)  # (height, width)
square_size = 0.025  # in meters

# Create a ZED camera object
zed = sl.Camera()

# Set initialization parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution
init_params.camera_fps = 30  # Set FPS
init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth for calibration

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Create an OpenCV window
cv2.namedWindow("Checkerboard", cv2.WINDOW_NORMAL)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., assuming checkerboard is fixed on the z=0 plane.
objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : checkerboard_size[1], 0 : checkerboard_size[0]].T.reshape(
    -1, 2
)
objp *= square_size

# Arrays to store object points and image points from all images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Capture loop
runtime_parameters = sl.RuntimeParameters()
while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image_zed = sl.Mat()
        zed.retrieve_image(
            image_zed, sl.VIEW.LEFT
        )  # Get the left image from the camera

        frame = image_zed.get_data()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)

        cv2.imshow("Checkerboard", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

zed.close()
cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Assuming you want to use the first image to define the world frame
if ret:
    # The rotation and translation vectors from the first image are used
    rvec, tvec = rvecs[0], tvecs[0]

    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # The transformation matrix from the world to the camera frame
    T_camera_world = np.eye(4)
    T_camera_world[:3, :3] = R
    T_camera_world[:3, 3] = tvec.flatten()

    # The transformation matrix from the camera to the world frame (inverse of T_camera_world)
    T_world_camera = np.linalg.inv(T_camera_world)

    # Now T_world_camera is the transformation from the camera frame to the checkerboard (world) frame
    print("Transformation matrix from the camera frame to the world frame:")
    print(T_world_camera)
