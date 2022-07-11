import cv2
import argparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
import open3d as o3d
import os
import glob
import json

from utils.loader import mat_loader
from utils.cp_utils import weighted_image, DBSCAN_pcd, nearset_point_of_two_lines, ray_marching_by_pose, heatmap
from utils.point_cloud import SampleManager


def main(args):

    root = args.root
    input_dir = os.path.join(root, args.in_dir)
    output_dir = os.path.join(input_dir, args.out_dir) # will be the child directory of input dir
    output_subdirs = {
        'heatmap':'heatmap',
        'confidence':'confidence',
        'annotation_2D':'annotation_2D',
        'rgb':'rgb',
        'depth':'depth'
    }

    # check and config dir
    assert os.path.exists(root), f'input dir {root} not exists'
    assert os.path.exists(input_dir), f'input dir {input_dir} not exists'
    complete_mat_in_dirs = glob.glob(f'{input_dir}/*/mat')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for key in output_subdirs.keys():
        complete_output_subdir = os.path.join(output_dir, output_subdirs[key])
        os.makedirs(complete_output_subdir, exist_ok=True)

    # load raw data (mat data : rgb-d, intrinsic, extrinsic relative to apriltag coordinate, depth scale)
    mat_lists = []
    for complete_mat_in_dir in complete_mat_in_dirs:
        l = mat_loader(complete_mat_in_dir, depth_scale=8000, cx=0.092, cy=0.12, cz=0.003)
        mat_lists.extend(l)
    
    # for visualization
    model_3d_noised = []
    model_3d_denoised = []

    # load rgb to run TSDF fusion to get the final pcd (mesh)
    smpMng = SampleManager(complete_mat_in_dirs, depth_scale=8000, loader=mat_loader, cx=0.092, cy=0.12, cz=0.003)
    print('[ Running TSDF fusion to reconstruct the 3D model ... ]')
    mesh_model = smpMng.meshModel # voxel_length=0.002, sdf_trunc=0.01
    mesh_model.crop(pt1=(-0.15, -0.15, 0.01), pt2=(0.15, 0.15, 0.5))
    mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh_model.mesh)
    o3d.io.write_triangle_mesh("{}/mesh.stl".format(output_dir), mesh)

    mesh_model.visualize()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = mesh_model.mesh.vertices
    # pcd.colors = mesh_model.mesh.vertex_colors
    
    # # world coordinate
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # model_3d_noised.append(coordinate)
    # model_3d_denoised.append(coordinate)


    # # add 3D point cloud (mesh)
    # pcd_pts = np.array(pcd.points)
    # pcd_colors = np.array(pcd.colors)
    # model_3d_noised.append(pcd)
    
    # # add 3D denoised point cloud (mesh)
    # print('[ Computing the clustered point cloud ... ]')
    # clustered_ind = DBSCAN_pcd(pcd_pts, eps=0.002, min_samples=10, max_cluster=1)
    # clustered_pcd_pts = pcd_pts[clustered_ind]
    # clustered_pcd_colors = pcd_colors[clustered_ind]

    # clustered_pcd = o3d.geometry.PointCloud()
    # if args.remove_lightbar:
    #     clustered_pcd.points = o3d.utility.Vector3dVector(clustered_pcd_pts)
    #     clustered_pcd.colors = o3d.utility.Vector3dVector(clustered_pcd_colors)

    #     # load light bar
    #     light_bar_pcd = o3d.io.read_point_cloud(f'{root}/light_bar/pcd.ply')
    #     light_bar_pts = np.asarray(light_bar_pcd.points)
    #     print(light_bar_pts.shape)

    #     nn = NearestNeighbors(n_neighbors=1)
    #     nn.fit(clustered_pcd_pts)
    #     inds_remove = nn.kneighbors(light_bar_pts, return_distance=False)
    #     inds_mask = np.ones(len(clustered_pcd_pts), dtype=bool)
    #     inds_mask[inds_remove,] = False

    #     clustered_pcd = o3d.geometry.PointCloud()
    #     clustered_pcd.points = o3d.utility.Vector3dVector(clustered_pcd_pts[inds_mask])
    #     clustered_pcd.colors = o3d.utility.Vector3dVector(clustered_pcd_colors[inds_mask])
    #     print(f'[num of points before removing lignt bar : {len(clustered_pcd_pts)}]')
    #     print(f'[num of points after removing lignt bar : {len(clustered_pcd.points)}')

    #     model_3d_denoised.append(clustered_pcd)
    # else :
    #     clustered_pcd.points = o3d.utility.Vector3dVector(clustered_pcd_pts)
    #     clustered_pcd.colors = o3d.utility.Vector3dVector(clustered_pcd_colors)
    #     model_3d_denoised.append(clustered_pcd)


    # # save the result
    # clustered_pcd.estimate_normals()
    # distances = clustered_pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radious = 0.5 * avg_dist
    
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(clustered_pcd, o3d.utility.DoubleVector([radious, radious * 2]))
    # mesh.triangle_normals = o3d.utility.Vector3dVector([])
    
    # # visualize all the results
    # o3d.visualization.draw_geometries(model_3d_noised)
    # o3d.visualization.draw_geometries(model_3d_denoised)
    # o3d.visualization.draw_geometries([mesh])
    
    # o3d.io.write_triangle_mesh("{}/mesh.stl".format(output_dir), mesh)
    # o3d.io.write_point_cloud("{}/mesh.ply".format(output_dir), clustered_pcd)
    # print("{}/mesh.obj saved".format(output_dir))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='cam_output', type=str)
    parser.add_argument('--in-dir', '-id', default='', type=str)
    parser.add_argument('--out-dir', '-od', default='annotation', type=str)
    parser.add_argument('--remove-lightbar', '-rlb', action='store_true')
    args = parser.parse_args()
    main(args)