import os
import numpy as np
import matplotlib.pyplot as plt
import cam_params
from scipy.spatial.transform import Rotation

def rotmat_to_RPY_Ros(matrix):
    # Convert the rotation matrix to a Rotation object
    rotation = Rotation.from_matrix(matrix)

    # Get Euler angles in ZYX convention
    yaw, pitch, roll = rotation.as_euler('ZYX', degrees=False)

    return roll, pitch, yaw

info_folder = "/home/sopho/Videos/Rec_8/calib/info"  # Replace with the path to your info folder

# Load extrinsics
extrinsics = cam_params.load_extrinsics(info_folder)

first_camera = True
for cam_idx, extrinsic_data in extrinsics.items():
    (R_l, T_l), (R_r, T_r) = extrinsic_data

(R_l_1, T_l_1), (R_r_1, T_r_1) = extrinsics["1"]
(R_l_0, T_l_0), (R_r_0, T_r_0) = extrinsics["0"]

#define the coordinate frame transform
Trans_0_l, Trans_0_r = np.eye(4), np.eye(4)
Trans_0_l[:3, :3], Trans_0_r[:3, :3]  = R_l_0, R_r_0
Trans_0_l[:3, 3], Trans_0_r[:3, 3] = T_l_0, T_r_0
Trans_0_l, Trans_0_r = np.linalg.inv(Trans_0_l), np.linalg.inv(Trans_0_r)

Trans_1_l, Trans_1_r = np.eye(4), np.eye(4)
Trans_1_l[:3, :3], Trans_1_r[:3, :3]  = R_l_1, R_r_1
Trans_1_l[:3, 3], Trans_1_r[:3, 3] = T_l_1, T_r_1
Trans_1_l, Trans_1_r = np.linalg.inv(Trans_1_l), np.linalg.inv(Trans_1_r)

#define transform into ros frame
R_ros = np.eye(3)
R_ros[0,:]=[0,0,1]
R_ros[1,:]=[-1,0,0]
R_ros[2,:]=[0,-1,0]
R_ros= np.linalg.inv(R_ros)
T_ros=np.zeros(3)
Trans_ros = np.eye(4)
Trans_ros[:3, :3] = R_ros
Trans_ros[:3, 3] = T_ros

#transform into ros coordinate frame

Trans_0_l, Trans_0_r = Trans_0_l @ Trans_ros, Trans_0_r @ Trans_ros
Trans_1_l, Trans_1_r = Trans_1_l @ Trans_ros, Trans_1_r @ Trans_ros

left_lens2base_link_t = np.array([0.01, -0.06, -0.015])
Trans_left2base = np.eye(4)
Trans_left2base[:3,3] = left_lens2base_link_t

base_0 = Trans_0_l.copy()
base_0 = base_0 @ Trans_left2base

base_1 = Trans_1_l.copy()
base_1 = base_1 @ Trans_left2base

# define transform from robot to chessboard
T_robot_chess = [0.358, 0.03, 0.006]
R_robot_chess = np.eye(3)
R_robot_chess[0,:]=[-1,0,0]
R_robot_chess[1,:]=[0,1,0]
R_robot_chess[2,:]=[0,0,-1]
Trans_robot_chess = np.eye(4)
Trans_robot_chess[:3, :3] = R_robot_chess
Trans_robot_chess[:3, 3] = T_robot_chess

#transform into robot / world coordinate frame
Trans_A_final = Trans_robot_chess @ base_0
Trans_B_final = Trans_robot_chess @ base_1

r0, p0, y0 = rotmat_to_RPY_Ros(Trans_A_final[:3, :3])
r1, p1, y1 = rotmat_to_RPY_Ros(Trans_B_final[:3, :3])

#print the values to input into the launchfile
print(f"The translation from robot to baselink 0 is: \n {Trans_A_final[:3,3]}")
print(f"The translation from robot to baselink 1 is: \n {Trans_B_final[:3,3]}")
print(f"The RPY angles from robot to baselink 0 is: \n {r0,p0,y0}")
print(f"The RPY angles from from robot to baselink 1 is: \n {r1,p1,y1}")




