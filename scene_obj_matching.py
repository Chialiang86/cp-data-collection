import argparse
import os 
import glob
import json
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import numpy as np

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

    return transform

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

def main(args):
    scene_pcd_f = args.scene_pcd
    scene_json_f = args.scene_json
    obj_pcd_f = args.obj_pcd
    obj_json_f = args.obj_json

    if not os.path.exists(scene_pcd_f):
        print(f'{scene_pcd_f} not exists')
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

    with open(scene_json_f, 'r') as f1:
        scene_pcd_3kpt = json.load(f1)
    with open(obj_json_f, 'r') as f2:
        obj_pcd_3kpt = json.load(f2)

    get_transform_from_3kpt(scene_pcd_3kpt, obj_pcd_3kpt)

    scene_pcd = o3d.io.read_point_cloud(scene_pcd_f)
    obj_pcd = o3d.io.read_point_cloud(obj_pcd_f)
    colors = np.zeros(np.asarray(obj_pcd.points).shape)
    colors[:, 0] = 1.0
    obj_pcd.colors = o3d.utility.Vector3dVector(colors)

    # before alignment
    o3d.visualization.draw_geometries([scene_pcd, obj_pcd])
    
    init_transform_os = get_transform_from_3kpt(scene_pcd_3kpt, obj_pcd_3kpt) # obj_pcd_3kpt -> scene_pcd_3kpt
    transform_os = open3d_icp_algorithm(obj_pcd, scene_pcd, init_trans=init_transform_os, thresh=0.01, iteration=1000) # obj_pcd_3kpt -> scene_pcd_3kpt
    obj_pcd.transform(init_transform_os)

    # after alignment
    o3d.visualization.draw_geometries([scene_pcd, obj_pcd])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_pcd', '-sp', type=str, default='', help='the first point cloud (.ply)')
    parser.add_argument('--obj_pcd', '-op', type=str, default='', help='the second point cloud (.ply)')
    parser.add_argument('--scene_json', '-sj', type=str, default='', help='the first point cloud 3-point matching info (.json)')
    parser.add_argument('--obj_json', '-oj', type=str, default='', help='the second point cloud 3-point matching info (.json)')
    args = parser.parse_args()
    main(args)
