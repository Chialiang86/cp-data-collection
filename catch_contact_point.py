import cv2
import argparse
import numpy as np
import open3d as o3d
import os

from utils.loader import mat_loader
from utils.cp_utils import weighted_image, DBSCAN_pcd, nearset_point_of_two_lines, ray_marching_by_pose, heatmap


def main(args):

    mat_input_dir = args.mat_input_dir
    pcd_input_dir = args.pcd_input_dir
    obj_name = args.obj
    output_dir = args.output_dir

    # check dir
    assert os.path.exists(mat_input_dir), f'input dir {mat_input_dir} not exists'
    assert os.path.exists(pcd_input_dir), f'input dir {pcd_input_dir} not exists'
    assert os.path.exists(output_dir), f'input dir {output_dir} not exists'
    complete_mat_in_dir = os.path.join(mat_input_dir, obj_name)
    complete_pcd_in_file = os.path.join(pcd_input_dir, f'{obj_name}/{obj_name}.ply')
    assert os.path.exists(complete_mat_in_dir), f'input dir {complete_mat_in_dir} not exists'
    assert os.path.exists(complete_pcd_in_file), f'input dir {complete_pcd_in_file} not exists'
    
    # config out dir
    complete_out_dir = os.path.join(output_dir, obj_name)
    if not os.path.exists(complete_out_dir):
        os.mkdir(complete_out_dir)

    mat_lists = mat_loader(complete_mat_in_dir, depth_scale=8000, cx=0.092, cy=0.12, cz=0.003)

    # for visualization
    model_3d_noised = []
    model_3d_denoised = []

    # for ray marching and contact points candidate
    line_pts = []
    line_ids = []
    line_vec = []
    
    for i, sample in enumerate(mat_lists):
        print('processing ', sample['spath'])
        print('processing ', sample.keys())

        color = sample['color'] # rgb
        depth = sample['depth']
        pose = sample['pose']
        intrinsic = sample['intr']
        print(intrinsic)
        # print(np.max(depth), sample['dscale'])

        file_prefix = os.path.splitext(sample['spath'])[0].split('/')[-1]
        output_heatmap_path = os.path.join(complete_out_dir, f'{file_prefix}_heatmap.png')

        # save heatmap
        weighted_color = weighted_image(color)
        (condx, condy) = np.where(weighted_color == np.max(weighted_color))
        max_xy = np.vstack((condx, condy)).T
        conf = heatmap(weighted_color.shape[:2], max_xy, 150)
        conf_heatmap = cv2.addWeighted(color, 0.3, conf, 0.7, 0)

        cv2.imwrite(output_heatmap_path, conf_heatmap)
        print(f'{output_heatmap_path} saved')

        # ray marching
        start, end = ray_marching_by_pose(np.linalg.inv(pose), intrinsic, max_xy, 0.45)
        line_pts.extend([start, end]) # line segment end points
        line_ids.append([ 2 * i, 2 * i + 1]) # line segment end points index
        line_vec.append(end - start) # line vector

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coordinate.transform(np.linalg.inv(pose))
        # model_3d_noised.append(coordinate)
        # model_3d_denoised.append(coordinate)

        break
    
    # draw ray matching result
    line_colors = np.asarray([[1, 0, 0] for i in range(len(line_pts))])
    line_set = o3d.geometry.LineSet()
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    line_set.points = o3d.utility.Vector3dVector(line_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_ids)
    # model_3d_noised.append(line_set)
    # model_3d_denoised.append(line_set)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    model_3d_noised.append(coordinate)
    model_3d_denoised.append(coordinate)

    # get contact point candidate
    candidate_pts = []
    for i in range(len(line_vec)):
        for j in range(i + 1, len(line_vec)):
            if i != j:
                x1, x2 = nearset_point_of_two_lines(line_pts[i * 2], line_vec[i], line_pts[j * 2], line_vec[j])
                candidate_pts.append(x1)
                candidate_pts.append(x2)
    candidate_pts = np.array(candidate_pts)
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_pts)
    candidate_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(line_pts))])
    model_3d_noised.append(candidate_pcd)

    # runing DBSCAN
    clustering_ind = DBSCAN_pcd(candidate_pts, eps=0.001, min_samples=10, max_cluster=2)
    clustering_pts = candidate_pts[clustering_ind]
    clustering_colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(clustering_ind))])
    clustering_pcd = o3d.geometry.PointCloud()
    clustering_pcd.points = o3d.utility.Vector3dVector(clustering_pts)
    clustering_pcd.colors = o3d.utility.Vector3dVector(clustering_colors)
    model_3d_denoised.append(clustering_pcd)

    # add 3D shape
    pcd = o3d.io.read_point_cloud(complete_pcd_in_file)
    pcd_pts = np.array(pcd.points)
    pcd_colors = np.array(pcd.colors)
    clustering_ind = DBSCAN_pcd(pcd_pts, eps=0.002, min_samples=10, max_cluster=1)
    clustering_pcd_pts = pcd_pts[clustering_ind]
    clustering_pcd_colors = pcd_colors[clustering_ind]
    clustering_pcd = o3d.geometry.PointCloud()
    clustering_pcd.points = o3d.utility.Vector3dVector(clustering_pcd_pts)
    clustering_pcd.colors = o3d.utility.Vector3dVector(clustering_pcd_colors)
    model_3d_noised.append(pcd)
    model_3d_denoised.append(clustering_pcd)
    o3d.visualization.draw_geometries(model_3d_noised)
    o3d.visualization.draw_geometries(model_3d_denoised)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat-input-dir', '-mid', default='viewer_out', type=str)
    parser.add_argument('--pcd-input-dir', '-pid', default='mesh_model', type=str)
    parser.add_argument('--obj', '-o', default='scissor', type=str)
    parser.add_argument('--output-dir', '-od', default='contact_point', type=str)
    args = parser.parse_args()
    main(args)