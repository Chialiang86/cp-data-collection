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
    mesh_model.crop(pt1=(-0.15, -0.15, 0.001), pt2=(0.15, 0.15, 0.5))

    mesh_model.visualize()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh_model.mesh.vertices
    pcd.colors = mesh_model.mesh.vertex_colors
    if len(pcd.points) == 0:
        print(f'[error : no points in point cloud]')
        return
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='cam_output', type=str)
    parser.add_argument('--in-dir', '-id', default='', type=str)
    parser.add_argument('--out-dir', '-od', default='annotation', type=str)
    parser.add_argument('--remove-lightbar', '-rlb', action='store_true')
    args = parser.parse_args()
    main(args)