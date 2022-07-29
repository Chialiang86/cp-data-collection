import argparse
import os 
import glob
import json
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from PIL import Image
import numpy as np
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

def render(pcd, extr : np.ndarray, intr : np.ndarray, path : str, duration=0) -> np.ndarray:
    assert extr.shape == (4, 4)
    assert intr.shape == (6,)

    width, height, fx, fy, cx, cy = intr

    # Visualize Point Cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(width), height=int(height), visible=True)
    vis.add_geometry(pcd)

    # Read camera params
    param = o3d.camera.PinholeCameraParameters()

    param.extrinsic = extr
    
    o3d_intr = o3d.camera.PinholeCameraIntrinsic()
    o3d_intr.set_intrinsics(int(width), int(height), fx, fy, cx, cy)
    param.intrinsic = o3d_intr

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)

    # print("{}\n"
    #           "{}".format(param.extrinsic, param.intrinsic.intrinsic_matrix))

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
    scene_pcd_dir = args.scene_pcd
    scene_json_f = args.scene_json
    obj_pcd_f = args.obj_pcd
    obj_json_f = args.obj_json

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

    scene_pcd_fs = glob.glob(f'{scene_pcd_dir}/*.ply')
    assert len(scene_pcd_fs) > 0, f'no pcd files in {scene_pcd_dir}'
    scene_pcd_fs.sort(key=lambda s: int(os.path.splitext(s)[0].split('_')[-1]))

    # # first frame obj-scene alignment
    # get_transform_from_3kpt(scene_pcd_3kpt, obj_pcd_3kpt)

    first_scene_pcd = o3d.io.read_point_cloud(scene_pcd_fs[0])
    first_obj_pcd = o3d.io.read_point_cloud(obj_pcd_f)
    colors = np.zeros(np.asarray(first_obj_pcd.points).shape)
    colors[:, 0] = 1.0
    first_obj_pcd.colors = o3d.utility.Vector3dVector(colors)

    # # before alignment
    # o3d.visualization.draw_geometries([scene_pcd, obj_pcd])
    
    init_transform_os, scene_kpts, obj_kpts = get_transform_from_3kpt(scene_pcd_3kpt, obj_pcd_3kpt) # obj_pcd_3kpt -> scene_pcd_3kpt
    transform_os = align_obj_to_scene(first_obj_pcd, first_scene_pcd, init_trans=init_transform_os, thresh=0.01, iteration=1000) # obj_pcd_3kpt -> scene_pcd_3kpt
    first_obj_pcd.transform(init_transform_os)

    first_kpt = np.ones(4)
    first_kpt[:3] = obj_kpts[:, 0]
    first_kpt = first_kpt @ init_transform_os.T

    # o3d.visualization.draw_geometries([first_scene_pcd, first_obj_pcd])

    obj_pcds = [first_obj_pcd]

    render_extr_right = np.array([[1.0000000,  0.0000000,  0.0000000,    0.3],
                            [0.0000000,  0.5000000,  0.8660254,    -1],
                            [0.0000000, -0.8660254,  0.5000000,    0.6],
                            [0.0000000,  0.0000000,  0.0000000,    1.]])
    render_extr_top = np.array([[1.0000000,  0.0000000,  0.0000000,    0.],
                            [0.0000000,  1.0000000,  0.0000000,    0.],
                            [0.0000000,  0.0000000,  1.0000000,    0.8],
                            [0.0000000,  0.0000000,  0.0000000,    1.]])
    render_intr = np.array([1080, 810, 616.35845947, 616.98779297, 539.5, 404.5])

    # thresh = 0.01
    # global_threshold = 1.5 * thresh
    # local_threshold = 0.5 * thresh

    frames_right = []
    frames_top = []
    line_points = [first_kpt]
    print(line_points)
    merge_point_cloud = None
    for i in range(1, len(scene_pcd_fs) - 60):
        print(f'processing {scene_pcd_fs[i]}')

        tmp_obj_pcd = obj_pcds[-1] # get last frame
        tmp_scene_pcd = o3d.io.read_point_cloud(scene_pcd_fs[i])
        transform_os = align_obj_to_scene(tmp_obj_pcd, tmp_scene_pcd, init_trans=np.identity(4), thresh=0.01, iteration=1000) # obj_pcd_3kpt -> scene_pcd_3kpt
        tmp_obj_pcd.transform(transform_os)
        obj_pcds.append(tmp_obj_pcd)

        tmp_obj_kpt = line_points[-1] @ transform_os.T
        line_points.append(tmp_obj_kpt)

        file_prefix = os.path.splitext(scene_pcd_fs[i])[0].split('/')[-1]

        merge_point_cloud = o3d.geometry.PointCloud()
        merge_point_cloud_points_arr = np.vstack((np.asarray(tmp_scene_pcd.points), np.asarray(tmp_obj_pcd.points)))
        merge_point_cloud.points = o3d.utility.Vector3dVector(merge_point_cloud_points_arr)
        merge_point_cloud_colors_arr = np.vstack((np.asarray(tmp_scene_pcd.colors), colors))
        merge_point_cloud.colors = o3d.utility.Vector3dVector(merge_point_cloud_colors_arr)

        tmp_frame_right = render(merge_point_cloud, render_extr_right, render_intr, path=f'{render_out_dir}/{file_prefix}.jpg', duration=0)
        frames_right.append(tmp_frame_right)
        tmp_frame_top = render(merge_point_cloud, render_extr_top, render_intr, path=f'{render_out_dir}/{file_prefix}.jpg', duration=0)
        frames_top.append(tmp_frame_top)
        # o3d.visualization.draw_geometries([tmp_scene_pcd, tmp_obj_pcd])


    imgs_right = [Image.fromarray(frame) for frame in frames_right]
    imgs_top = [Image.fromarray(frame) for frame in frames_top]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs_right[0].save(f'{render_out_dir}/render_right.gif', save_all=True, append_images=imgs_right[1:], duration=50, loop=0)
    imgs_top[0].save(f'{render_out_dir}/render_top.gif', save_all=True, append_images=imgs_top[1:], duration=50, loop=0)

    # draw lines
    print(np.array(line_points)[:, :3])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points)[:, :3])
    line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(len(line_points)-1)]))
    line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for i in range(len(line_points)-1)]))

    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Keypoint Trajectory")
    # vis.get_render_option().line_width = 10.0
    # vis.add_geometry(merge_point_cloud)
    # vis.add_geometry(line_set)	    

    # while True:
    #     vis.poll_events()
    #     vis.update_renderer()
    o3d.visualization.draw_geometries([merge_point_cloud, line_set])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_pcd', '-sp', type=str, default='', help='the first point cloud (.ply)')
    parser.add_argument('--obj_pcd', '-op', type=str, default='', help='the second point cloud (.ply)')
    parser.add_argument('--scene_json', '-sj', type=str, default='', help='the first point cloud 3-point matching info (.json)')
    parser.add_argument('--obj_json', '-oj', type=str, default='', help='the second point cloud 3-point matching info (.json)')
    args = parser.parse_args()
    main(args)
