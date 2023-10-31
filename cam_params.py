import os
import pickle


def load_intrinsics(info_folder):
    with open(os.path.join(info_folder, "intrinsics.txt"), "rb") as f:
        data = f.read()
    intrinsics = pickle.loads(data)
    return intrinsics


def load_extrinsics(info_folder):
    with open(os.path.join(info_folder, "extrinsics.txt"), "rb") as f:
        data = f.read()
    extrinsics = pickle.loads(data)
    return extrinsics

def load_img_size(info_folder):
    with open(os.path.join(info_folder, "img_size.txt"), "rb") as f:
        data = f.read()
    img_size = pickle.loads(data)
    return img_size

def save_intrinsics(intrinsics, disto, cam_idxs, info_folder):
    # 0: left, 1: right
    intrinsics = {cam_idx: ([K[0], D[0]], [K[1], D[1]]) for K, D, cam_idx in zip(intrinsics, disto, cam_idxs)}
    serialized = pickle.dumps(intrinsics)
    with open(os.path.join(info_folder, "intrinsics.txt"), "wb") as f:
        f.write(serialized)


def save_extrinsics(r_vecs, t_vecs, cam_idxs, info_folder):
    extrinsics = {cam_idx: ([R[0], t[0]], [R[1], t[1]]) for R, t, cam_idx in zip(r_vecs, t_vecs, cam_idxs)}
    serialized = pickle.dumps(extrinsics)
    with open(os.path.join(info_folder, "extrinsics.txt"), "wb") as f:
        f.write(serialized)


def save_img_size(height, width, info_folder):
    img_size = [height, width]
    serialized = pickle.dumps(img_size)
    with open(os.path.join(info_folder, "img_size.txt"), "wb") as f:
        f.write(serialized)