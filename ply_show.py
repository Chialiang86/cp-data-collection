import os
import numpy as np
import open3d as o3d
import argparse

from utils.cp_utils import weighted_pcd_colors, DBSCAN_pcd

def judge_size(pcd):
    points = np.asarray(pcd.points)
    max_height = np.max(points[:,2])
    print(max_height)

def main(args):
    file_path = '{}/{}.ply'.format(args.obj, args.obj)
    mesh_path = os.path.join(args.root, file_path)
    if not os.path.exists(mesh_path):
        raise BaseException('%s not exists.' % mesh_path)

    print("Load a ply point cloud, print it, and render it")
    model_3d = []
    model_denoised_3d = []


    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    pcd = o3d.io.read_point_cloud(mesh_path)
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    pcd_colors_weighted = weighted_pcd_colors(pcd_colors)
    max_ind = np.where(pcd_colors_weighted == np.max(pcd_colors_weighted))[0]

    coordinate_cp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    coordinate_cp.translate(pcd_points[max_ind].T)

    dbscan_ind = DBSCAN_pcd(pcd_points, 0.002, 10, 1)
    pcd_denoised = o3d.geometry.PointCloud()
    pcd_denoised_points = pcd_points[dbscan_ind]
    pcd_denoised_colors = pcd_colors[dbscan_ind]
    pcd_denoised.points = o3d.utility.Vector3dVector(pcd_denoised_points)
    pcd_denoised.colors = o3d.utility.Vector3dVector(pcd_denoised_colors)

    pcd_colors_denoised_weight = weighted_pcd_colors(pcd_denoised_colors)
    max_denoised_ind = np.where(pcd_colors_denoised_weight == np.max(pcd_colors_denoised_weight))[0]

    coordinate_denoised_cp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    coordinate_denoised_cp.translate(pcd_denoised_points[max_denoised_ind].T)
    
    model_3d = [coordinate, coordinate_cp, pcd]
    model_denoised_3d = [coordinate, coordinate_cp, pcd_denoised]

    o3d.visualization.draw_geometries(model_3d)
    o3d.visualization.draw_geometries(model_denoised_3d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default='mesh_model/')
    parser.add_argument('-o', '--obj', default='scissor', type=str)
    args = parser.parse_args()
    main(args)