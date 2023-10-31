import math
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_to_euler_zyx(matrix):
    # Convert the rotation matrix to a Rotation object
    rotation = Rotation.from_matrix(matrix)

    # Get Euler angles in ZYX convention
    yaw, pitch, roll = rotation.as_euler('ZYX', degrees=False)

    return roll, pitch, yaw

def rotmat_to_RPY_Ros(matrix):
    pitch = np.arcsin(matrix[0, 2])

    # Calculate yaw (rotation around X axis)
    if np.cos(pitch) != 0:
        roll = np.arctan2(-matrix[1, 2] / np.cos(pitch), matrix[2, 2] / np.cos(pitch))
    else:
        roll = 0

    # Calculate roll (rotation around Z axis)
    if np.cos(pitch) != 0:
        yaw = np.arctan2(-matrix[0, 1] / np.cos(pitch), matrix[0, 0] / np.cos(pitch))
    else:
        yaw = 0

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


# define roll, pitch, yaw
r = 44.7837306
p = -32.30176911
y = -65.97820131

# construct rotation matrices
cosr, cosp, cosy = math.cos(np.deg2rad(r)), math.cos(np.deg2rad(p)), math.cos(np.deg2rad(y))
sinr, sinp, siny = math.sin(np.deg2rad(r)), math.sin(np.deg2rad(p)), math.sin(np.deg2rad(y))

R_x = np.array([[1, 0, 0], [0, cosr, -sinr], [0, sinr, cosr]])
R_y = np.array([[cosp, 0, sinp], [0, 1, 0], [-sinp, 0, cosp]])
R_z = np.array([[cosy, -siny, 0], [siny, cosy, 0], [0, 0, 1]])

# print ROS RPY_extrinsic --> Rotate around global X the global Y then global Z axis

R_rpy_extrinsic = R_x @ R_y @ R_z

print(np.round(R_rpy_extrinsic, 3))
print("\n")

# a,b,c = rotation_matrix_to_euler_angles_xyz(R_rpy_extrinsic)
a, b, c = rotmat_to_RPY_Ros(R_rpy_extrinsic)


Rot_test = np.array([[ 0.344086, 0.77203918, -0.53437845],[-0.80153819, -0.05489451, -0.59541845], [-0.48902082,  0.63319989,  0.59993045]])

roll, pitch, yaw = rotation_matrix_to_euler_zyx(Rot_test)
print(Rot_test)
print("\n")
RPY = [roll,pitch,yaw]
print(RPY)
