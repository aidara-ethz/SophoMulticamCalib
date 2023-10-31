import pprint

import numpy as np

import cam_params
import os
import pickle
import numpy



def load_extrinsics(info_folder):
    with open(os.path.join(info_folder, "extrinsics.txt"), "rb") as f:
        data = f.read()
    extrinsics = pickle.loads(data)
    return extrinsics

def inverse_transform(R, t):
    T_inv = np.eye(4, 4)
    T_inv[:3, :3] = np.transpose(R)
    T_inv[:3, 3] = -1 * np.matmul(np.transpose(R), (t))
    return T_inv


info_folder = "/home/sopho/Videos/Rec_9/calib/info"  # Replace with the path to your info folder
extrinsics = load_extrinsics(info_folder)
with open('extrinsics_hand_tracking.txt', 'w') as file:
    pprint.pprint(extrinsics, stream=file)

(R_l_1, T_l_1), (R_r_1, T_r_1) = extrinsics["1"]
(R_l_0, T_l_0), (R_r_0, T_r_0) = extrinsics["0"]

print("all transforms in camera frame to checkerboard frame i.e. the translation is c_T_cs with c as camera frame and "
      "s as checkerboard frame")
#print("Rotation l1 ", R_l_1)
print("Translation l1 ", T_l_1)
#print("Rotation l0 ", R_l_0)
print("Translation l0 ", T_l_0)
print("Translation r0 ", T_r_0)
print("Translation r1 ", T_r_1)
print("diff left ", T_l_1-T_l_0)
print("diff right ", T_r_1-T_r_0)

T_sc_l1 = inverse_transform(R_l_1, T_l_1)
T_sc_l0 = inverse_transform(R_l_0, T_l_0)
T_sc_r1 = inverse_transform(R_r_1, T_r_1)
T_sc_r0 = inverse_transform(R_r_0, T_r_0)

# ToDo: Include camera serial number in file

print("T_sc l1 \n", T_sc_l1)
print("T_sc l0 \n", T_sc_l0)
print("T_sc r1 \n", T_sc_r1)
print("T_sc r0 \n", T_sc_r0)




with open('extrinsics_readable.pkl', 'wb') as f:
    pickle.dump(T_sc_l1, f)


