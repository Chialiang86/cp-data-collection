import argparse
import os 
import glob
import json
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import open3d as o3d
from PIL import Image
import numpy as np
import cv2
import time


def get_transform_from_3kpt(scene_pcd_3kpt : dict, obj_pcd_3kpt : dict):
    
    scene_points = np.zeros((3, 3))
    obj_points = np.zeros((3, 3))

    for item in scene_pcd_3kpt['keypoints']:
        scene_points[:, item['class_id']] = item['position']
    for item in obj_pcd_3kpt['keypoints']:
        obj_points[:, item['class_id']] = item['position']
    
    scene_mean = np.mean(scene_points, axis=1).reshape((3, 1))
    obj_mean = np.mean(obj_points, axis=1).reshape((3, 1))

    scene_points_ = scene_points - scene_mean
    obj_points_ = obj_points - obj_mean

    W = scene_points_ @ obj_points_.T
    
    u, s, vh = np.linalg.svd(W, full_matrices=True)
    R = u @ vh
    t = scene_mean - R @ obj_mean

    transform = np.identity(4)
    transform[:3,:3] = R
    transform[:3, 3] = t.reshape((1, 3))

    return transform, scene_points, obj_points

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def open3d_icp_algorithm(pc1, pc2, init_trans=np.identity(4), thresh=0.001, iteration=1000):
    pc1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    icp_fine = o3d.pipelines.registration.registration_icp(
        pc1, pc2, thresh,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))
    transformation_icp = icp_fine.transformation
    return transformation_icp

def align_obj_to_scene(obj_pcd, scene_pcd, init_trans=np.identity(4), thresh=0.01, iteration=1000):
    # # down sample and get features
    # src_down, src_fpfh = preprocess_point_cloud(tmp_obj_pcd, thresh)
    # dst_down, dst_fpfh = preprocess_point_cloud(tmp_scene_pcd, thresh)

    # # global registration
    # icp_rough = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     src_down, dst_down, src_fpfh, dst_fpfh, True,
    #     global_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     3, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
    #             0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    #             global_threshold)
    #     ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    # tmp_obj_pcd.transform(icp_rough.transformation)
    # o3d.visualization.draw_geometries([tmp_scene_pcd, tmp_obj_pcd])
    # transform_os = open3d_icp_algorithm(tmp_obj_pcd, tmp_scene_pcd, init_trans=icp_rough.transformation, thresh=local_threshold, iteration=1000) # obj_pcd_3kpt -> scene_pcd_3kpt

    transform_os = open3d_icp_algorithm(obj_pcd, scene_pcd, init_trans=init_trans, thresh=thresh, iteration=iteration)
    return transform_os

def render_cv2(pcd : o3d.geometry.PointCloud, extr : np.ndarray, intr : np.ndarray, img_size : tuple,  distorsion = None, rgb = [0, 255, 0]):
    assert extr.shape == (4, 4)
    assert intr.shape == (3, 3)

    r_vec = extr[:3, :3]
    t_vec = extr[:3, 3]

    obj_pts = np.asarray(pcd.points)

    imgpts, jac = cv2.projectPoints(obj_pts, r_vec, t_vec, intr, distorsion)
    imgpts = np.rint(np.squeeze(imgpts)).astype(np.int32)

    W, H = img_size
    cond = np.where((imgpts[:, 1] < 0) | (imgpts[:, 1] >= H) | (imgpts[:, 0] < 0) | (imgpts[:, 0] >= W))
    imgpts = np.delete(imgpts, cond, axis=0)

    render_img = np.zeros((H, W, 3), dtype=np.uint8)
    render_img[imgpts[:, 1], imgpts[:, 0]] = rgb
    return render_img

def render(pcd : o3d.geometry.PointCloud, extr : np.ndarray, intr : np.ndarray, img_size : tuple, duration=0) -> np.ndarray:
    assert extr.shape == (4, 4)
    assert intr.shape == (3, 3)

    width, height = img_size
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], width / 2.0 - 0.5, height / 2.0 - 0.5

    # Visualize Point Cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(width), height=int(height), visible=True) # neet to be true
    vis.add_geometry(pcd)

    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 3.0

    # Read camera params
    param = o3d.camera.PinholeCameraParameters()

    param.extrinsic = extr
    
    o3d_intr = o3d.camera.PinholeCameraIntrinsic()
    o3d_intr.set_intrinsics(int(width), int(height), fx, fy, cx, cy)
    param.intrinsic = o3d_intr

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)

    # Updates
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    time.sleep(duration)
    # vis.capture_screen_image(path)
    image = vis.capture_screen_float_buffer()

    # Close
    vis.destroy_window()
    return (np.asarray(image) * 255).astype(np.uint8)

def main(args):
    scene_rgb_dir = args.scene_rgb
    scene_pcd_dir = args.scene_pcd
    scene_json_f = args.scene_json
    obj_pcd_f = args.obj_pcd
    obj_json_f = args.obj_json

    if not os.path.exists(scene_rgb_dir):
        print(f'{scene_rgb_dir} not exists')
        return
    if not os.path.exists(scene_pcd_dir):
        print(f'{scene_pcd_dir} not exists')
        return
    if not os.path.exists(obj_pcd_f):
        print(f'{obj_pcd_f} not exists')
        return
    if not os.path.exists(scene_json_f):
        print(f'{scene_json_f} not exists')
        return
    if not os.path.exists(obj_json_f):
        print(f'{obj_json_f} not exists')
        return
    
    render_out_dir = os.path.join(scene_pcd_dir, 'trajectory')
    os.makedirs(render_out_dir, exist_ok=True)

    with open(scene_json_f, 'r') as f1:
        scene_pcd_3kpt = json.load(f1)
    with open(obj_json_f, 'r') as f2:
        obj_pcd_3kpt = json.load(f2)


    # scene pcd
    scene_pcd_fs = glob.glob(f'{scene_pcd_dir}/*.ply')
    assert len(scene_pcd_fs) > 0, f'no pcd files in {scene_pcd_dir}'
    scene_pcd_fs.sort(key=lambda s: int(os.path.splitext(s)[0].split('_')[-1]))

    # first frame obj-scene alignment
    first_scene_pcd = o3d.io.read_point_cloud(scene_pcd_fs[0])
    first_obj_pcd = o3d.io.read_point_cloud(obj_pcd_f)
    colors = np.zeros(np.asarray(first_obj_pcd.points).shape)
    colors[:, 0] = 1.0
    first_obj_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # initial alignment
    init_transform_os, scene_kpts, obj_kpts = get_transform_from_3kpt(scene_pcd_3kpt, obj_pcd_3kpt) # obj_pcd_3kpt -> scene_pcd_3kpt
    # transform_os = align_obj_to_scene(first_obj_pcd, first_scene_pcd, init_trans=init_transform_os, thresh=0.01, iteration=1000) # obj_pcd_3kpt -> scene_pcd_3kpt
    first_obj_pcd.transform(init_transform_os)

    first_kpt = np.ones(4)
    first_kpt[:3] = obj_kpts[:, 0]
    first_kpt = first_kpt @ init_transform_os.T

    obj_pcds = [first_obj_pcd]

    render_extr_right = np.array([[1.0000000,  0.0000000,  0.0000000,    0.1],
                            [0.0000000,  0.5000000,  0.8660254,    -1],
                            [0.0000000, -0.8660254,  0.5000000,    0.8],
                            [0.0000000,  0.0000000,  0.0000000,    1.]])
    render_extr_top = np.array([[1.0000000,  0.0000000,  0.0000000,    0.],
                            [0.0000000,  1.0000000,  0.0000000,    0.],
                            [0.0000000,  0.0000000,  1.0000000,    0.5],
                            [0.0000000,  0.0000000,  0.0000000,    1.]])
    render_intr_mat = np.array([[616.35845947,            0, 539.5], 
                                [           0, 616.98779297, 404.5],
                                [           0,            0,      1]])
    render_img_shape = (1080, 810)

    render_extr_mats = [render_extr_right, render_extr_top]

    # thresh = 0.01  
    # global_threshold = 1.5 * thresh
    # local_threshold = 0.5 * thresh

    # scene mat
    scene_mat_dirs = glob.glob(f'{scene_rgb_dir}/*')
    scene_mat_files_dict = {}
    for scene_mat_dir in scene_mat_dirs:
        cam_id = scene_mat_dir.split('/')[-1]
        scene_mat_files = glob.glob(f'{scene_mat_dir}/mat_dynamic/*')
        scene_mat_files.sort(key=lambda s: int(os.path.splitext(s)[0].split('/')[-1])) # cam_output/20220727_172155_scissor/141722071222/mat_dynamic/31.mat
        scene_mat_files_dict[cam_id] = scene_mat_files

    # get render extrinsic
    extr_files = glob.glob(f'calibration/multicam/*.json')
    assert len(extr_files) > 0, 'no extrinsic given'
    extrinsic_dict = {}
    for extr_file in extr_files:
        with open(extr_file, 'r') as f:
            j_dict = json.load(f)
        extrinsic_dict[j_dict['master']] = np.identity(4) # will repeat 7 times
        extrinsic_dict[j_dict['slave']] = np.asarray(j_dict['extr_opt'])

    # get render intrinsic
    intr_files = glob.glob(f'calibration/intrinsic/*/intrinsic.npy')
    assert len(extr_files) > 0, 'no extrinsic given'
    intrinsic_dict = {}
    for intr_file in intr_files:
        intr_mat = np.load(intr_file)
        cam_id = intr_file.split('/')[-2] # ex : calibration/intrinsic/[cam_id]/intrinsic.npy
        intrinsic_dict[cam_id] = intr_mat
    
    # render image resolution
    img_shape = (640, 480)
    frames_dict = {}
    frames_3d_dict = [[] for i in range(len(render_extr_mats))]
    trajectory_dict = {}
    for key in intrinsic_dict.keys():
        frames_dict[key] = []
        trajectory_dict[key] = []
    
    line_points = []
    tmp_obj_kpt = first_kpt
    # merge_point_cloud = None
    for i in range(1, len(scene_pcd_fs)):
        print(f'processing {scene_pcd_fs[i]}')

        tmp_obj_pcd = obj_pcds[-1] # get last frame
        tmp_scene_pcd = o3d.io.read_point_cloud(scene_pcd_fs[i])
        transform_os = align_obj_to_scene(tmp_obj_pcd, tmp_scene_pcd, init_trans=np.identity(4), thresh=0.01, iteration=1000) # obj_pcd_3kpt -> scene_pcd_3kpt
        tmp_obj_pcd.transform(transform_os)
        obj_pcds.append(tmp_obj_pcd)

        tmp_obj_kpt = tmp_obj_kpt @ transform_os.T

        merge_point_cloud = o3d.geometry.PointCloud()
        merge_point_cloud_points_arr = np.vstack((np.asarray(tmp_scene_pcd.points), np.asarray(tmp_obj_pcd.points)))
        merge_point_cloud.points = o3d.utility.Vector3dVector(merge_point_cloud_points_arr)
        merge_point_cloud_colors_arr = np.vstack((np.asarray(tmp_scene_pcd.colors), colors))
        merge_point_cloud.colors = o3d.utility.Vector3dVector(merge_point_cloud_colors_arr)

        if i % 5 == 1:
            line_points.append(tmp_obj_kpt)
            for key in intrinsic_dict.keys():
                mat_path = scene_mat_files_dict[key][i]
                mat = sio.loadmat(mat_path)
                original_img = mat['color'][..., ::-1]

                intr = intrinsic_dict[key]
                extr = np.linalg.inv(extrinsic_dict[key])

                line_pt, jac = cv2.projectPoints(tmp_obj_kpt[:3], extr[:3, :3], extr[:3, 3], intr, None)
                line_pt = np.rint(np.squeeze(line_pt)).astype(np.int32)
                trajectory_dict[key].append(line_pt)

                render_image = render_cv2(tmp_obj_pcd, extr, intr, img_shape, distorsion=None, rgb=[255, 0, 0])
                weight = cv2.addWeighted(original_img, 1, render_image, 0.6, 0)
                frames_dict[key].append(weight)
            
            for j, render_extr_mat in enumerate(render_extr_mats):
                render_3d_image = render(merge_point_cloud, render_extr_mat, render_intr_mat, render_img_shape, duration=0)
                frames_3d_dict[j].append(render_3d_image)

    print("rendering trajectories ...")
    for key in tqdm(frames_dict.keys()):
        for i in range(len(frames_dict[key])):
            for j in range(len(trajectory_dict[key]) - 1, i + 1, -1):
                frames_dict[key][i] = cv2.line(frames_dict[key][i], trajectory_dict[key][j], trajectory_dict[key][j - 1], (163, 207, 190), thickness=2)

    # write rendered results back to original images
    print("saving gif files ...")
    for key in tqdm(frames_dict.keys()):
        frames = [Image.fromarray(frame) for frame in frames_dict[key]]
        frames.reverse()
        frames[0].save(f'{render_out_dir}/render_{key}.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)

    for i in range(len(frames_3d_dict)):
        frames_3d = [Image.fromarray(frame_3d) for frame_3d in frames_3d_dict[i]]
        frames_3d.reverse()
        frames_3d[0].save(f'{render_out_dir}/render_3d_{i}.gif', save_all=True, append_images=frames_3d[1:], duration=200, loop=0)

    # draw lines
    print(np.array(line_points)[:, :3])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points)[:, :3])
    line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(len(line_points)-1)]))
    line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for i in range(len(line_points)-1)]))

    o3d.visualization.draw_geometries([merge_point_cloud, line_set])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_rgb', '-si', type=str, default='', help='the scene images path')
    parser.add_argument('--scene_pcd', '-sp', type=str, default='', help='the first point cloud (.ply)')
    parser.add_argument('--obj_pcd', '-op', type=str, default='', help='the second point cloud (.ply)')
    parser.add_argument('--scene_json', '-sj', type=str, default='', help='the first point cloud 3-point matching info (.json)')
    parser.add_argument('--obj_json', '-oj', type=str, default='', help='the second point cloud 3-point matching info (.json)')
    args = parser.parse_args()
    main(args)
