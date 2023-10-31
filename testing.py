import numpy as np
import math


_NEXT_AXIS = [1, 2, 0, 1]

def euler_from_matrix(matrix, axes='sxyz'):
    firstaxis, parity, repetition, frame = 0, 1, 0, 0
    i, j, k = firstaxis, _NEXT_AXIS[firstaxis+parity], _NEXT_AXIS[firstaxis-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])

    if cy > 1e-6:
        ax = math.atan2( M[k, j],  M[k, k])
        ay = math.atan2(-M[k, i],  cy)
        az = math.atan2( M[j, i],  M[i, i])
    else:
        ax = math.atan2(-M[j, k],  M[j, j])
        ay = math.atan2(-M[k, i],  cy)
        az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az

    return ax, ay, az

def rotmat_to_fixed_euler(matrix):
    # The axes parameter 'rzyx' corresponds to the extrinsic XYZ convention
    roll, pitch, yaw = euler_from_matrix(matrix, axes='rzyx')
    return roll, pitch, yaw

# Example usage
R = np.array([
    [0.93629336, -0.27509585, 0.21835066],
    [0.28962948, 0.95642509, 0.03695701],
    [-0.19866933, 0.0978434, 0.97517033]
])

roll, pitch, yaw = rotmat_to_fixed_euler(R)
print("Roll: {:.2f}, Pitch: {:.2f}, Yaw: {:.2f}".format(roll, pitch, yaw))

points = np.array([
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0, 0, 0.1, 1]
]).T

print(f"{points[0, 0], points[1, 0], points[2, 0]} \n")
print(f"{points[0, :2], points[1, :2], points[2, :2]} \n")
print(f"{points[0, :3:2], points[1, :3:2], points[2, :3:2]} \n")
print(f"{points[0, :4:3], points[1, :4:3], points[2, :4:3]} \n")

r = -54.8689198
p = 53.2772388
y =  95.2674836
cosr = math.cos(np.deg2rad(r))
cosp = math.cos(np.deg2rad(p))
cosy = math.cos(np.deg2rad(y))
sinr = math.sin(np.deg2rad(r))
sinp = math.sin(np.deg2rad(p))
siny = math.sin(np.deg2rad(y))

Rot_z = np.array([[cosy,-siny,0],[siny,cosy,0],[0,0,1]])
Rot_y = np.array([[cosp,0,sinp],[0,1,0],[-sinp,0,cosp]])
Rot_x = np.array([[1,0,0],[0,cosr,-sinr],[0,sinr,cosr]])

Rot_tot = Rot_z @ (Rot_y @ Rot_x)
print(np.round(Rot_tot,3))
