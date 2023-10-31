import os
import pickle
import copy
import numpy as np
import open3d as o3d
import cv2


def draw_registration_result_original_color(source, target, transformation, write_path=None):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += source_temp
    merged_pcd += target
    if write_path:
        o3d.io.write_point_cloud(write_path, merged_pcd)

    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])


def create_pc_from_depth(intrinsics, extrinsics, rec_path, svo_recs, pc_merge_order):

    color_path = os.path.join(rec_path, "color")
    depth_path = os.path.join(rec_path, "depth")
    # intrinsics

    pcds = []
    for svo, (cam_id, (K, d)), (cam_id, (rot, trans)) in zip(svo_recs, intrinsics.items(), extrinsics.items()):
        assert svo.split('_')[1] == cam_id

        color_folder = os.path.join(color_path, svo.replace('.svo', ''))
        depth_folder = os.path.join(depth_path, svo.replace('.svo', ''))

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot
        extrinsics[:3, 3] = trans

        color_imgs = [os.path.join(color_folder, color_img) for color_img in os.listdir(color_folder)]
        depth_imgs = [os.path.join(depth_folder, depth_img) for depth_img in os.listdir(depth_folder)]
        for color_img, depth_img in zip(color_imgs, depth_imgs):
            color_image = cv2.imread(color_img, -1)
            color_image = color_image[:, :color_image.shape[1]//2, ::-1]
            color_image_left = o3d.geometry.Image(color_image.astype(np.uint8))

            depth_arr = cv2.imread(depth_img, -1)[:, :, 0]

            depth_image = o3d.geometry.Image(depth_arr.astype(np.float32))
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_left, depth_image,
                                                                          depth_scale=1000.0,
                                                                          depth_trunc=1.5,
                                                                          convert_rgb_to_intensity=False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(height=depth_arr.shape[0], width=depth_arr.shape[1],
                                                          fx=fx, fy=fy, cx=cx, cy=cy)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=intrinsic, extrinsic=extrinsics)
            #pcd.voxel_down_sample(0.04)
            pcd.uniform_down_sample(every_k_points=100)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=120, std_ratio=1.0)
            cl.estimate_normals()
            pcds.append(cl)
            break

    merged_pcd = o3d.geometry.PointCloud()
    for pc in pcds:
        merged_pcd += pc

    # save

    """
    current_transformation = np.identity(4)
    # icp
    color_icp = True
    if not color_icp:
        result_icp = o3d.pipelines.registration.registration_icp(pcds[0], pcds[1], 0.02, current_transformation,
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            pcds[0], pcds[1], 0.01, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=50))

    draw_registration_result_original_color(pcds[1], pcds[2], result_icp.transformation, os.path.join(rec_path, "icp.ply"))
    draw_registration_result_original_color(pcds[1], pcds[2], current_transformation, os.path.join(rec_path, "original.ply"))
    #o3d.visualization.draw_geometries([merged_pcd])
    """



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


if __name__ == '__main__':
    path = rf"D:\MVStereo\test_rec_3"
    calib_folder = os.path.join(path, "calib")
    info_folder = os.path.join(calib_folder, "info")

    rec_folder = [rec_folder for rec_folder in os.listdir(path) if rec_folder != "calib"][0]
    rec_folder = os.path.join(path, rec_folder)
    svo_files = [svo_file for svo_file in os.listdir(os.path.join(path, rec_folder)) if ".svo" in svo_file]

    intrinsics = load_intrinsics(info_folder)
    extrinsics = load_extrinsics(info_folder)

    pc_merge_order = [1, 2, 0]  # first merge pcs 1 & 2 and then 0
    create_pc_from_depth(intrinsics, extrinsics, rec_folder, svo_files, pc_merge_order)
