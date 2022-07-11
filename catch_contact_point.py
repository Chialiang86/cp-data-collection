import cv2
import argparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
import open3d as o3d
import os
import glob
import json

from utils.loader import mat_loader
from utils.cp_utils import weighted_image, DBSCAN_pcd, nearest_point_of_all_lines, ray_marching_by_pose, heatmap
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
    complete_mat_in_dirs = glob.glob(f'{input_dir}/*/mat_static')
    print(complete_mat_in_dirs)
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

    # for ray marching and contact points candidate
    line_pts = []
    line_ids = []
    line_vec = []

    # load rgb image, camera parameters and find the contact point, do ray marching, save image
    print('[ Loading mat files and running ray marching ... ]')
    for i, sample in enumerate(mat_lists):
        print('processing ', sample['spath'])

        color = sample['color'] # rgb
        depth = sample['depth']
        pose = sample['pose']
        intrinsic = sample['intr']
        # print(np.max(depth), sample['dscale'])
        
        # ex -> cam_output/0531-2/635206006429/mat/1.mat
        file_text = os.path.splitext(sample['spath'])[0]
        cam_id = file_text.split('/')[-3]
        img_id = file_text.split('/')[-1]
        file_prefix = f'{cam_id}_{img_id}' # get the final file name without extension

        # save heatmap
        weighted_color = weighted_image(color)
        (condx, condy) = np.where(weighted_color == np.max(weighted_color))
        max_xy = np.vstack((condx, condy)).T
        max_xy.astype(np.uint16)
        conf, rendered_conf = heatmap(weighted_color.shape[:2], max_xy, 150)
        conf_heatmap = cv2.addWeighted(color, 0.3, rendered_conf, 0.7, 0)

        # ray marching
        start, end = ray_marching_by_pose(np.linalg.inv(pose), intrinsic, max_xy, 1)
        line_pts.extend([start, end]) # line segment end points
        line_ids.append([ 2 * i, 2 * i + 1]) # the end points indexes of line segment
        line_vec.append(end - start) # line vector

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coordinate.transform(np.linalg.inv(pose))
        model_3d_noised.append(coordinate)
        model_3d_denoised.append(coordinate)

        # save to files
        output_heatmap_path = os.path.join(output_dir, output_subdirs['heatmap'], f'{file_prefix}_heatmap.png')
        cv2.imwrite(output_heatmap_path, conf_heatmap)
        print(f'{output_heatmap_path} saved')

        output_confidence_path = os.path.join(output_dir, output_subdirs['confidence'], f'{file_prefix}_confidence.npy')
        np.save(output_confidence_path, conf)
        print(f'{output_confidence_path} saved')
        
        output_annotation_path = os.path.join(output_dir, output_subdirs['annotation_2D'], f'{file_prefix}_annotation.json')
        json_dict = {
            'xy' : [
                [int(d[0]), int(d[1])] for d in max_xy
            ]
        }
        with open(output_annotation_path, 'w') as f:
            json.dump(json_dict, f)
        print(f'{output_annotation_path} saved')

        output_rgb_path = os.path.join(output_dir, output_subdirs['rgb'], f'{file_prefix}_rgb.png')
        cv2.imwrite(output_rgb_path, color)
        print(f'{output_rgb_path} saved')

        output_depth_path = os.path.join(output_dir, output_subdirs['depth'], f'{file_prefix}_depth.npy')
        np.save(output_depth_path, depth)
        print(f'{output_depth_path} saved')


    # load rgb to run TSDF fusion to get the final pcd (mesh)
    smpMng = SampleManager(complete_mat_in_dirs, depth_scale=8000, loader=mat_loader, cx=0.092, cy=0.12, cz=0.003)
    print('[ Running TSDF fusion to reconstruct the 3D model ... ]')
    mesh_model = smpMng.meshModel # voxel_length=0.002, sdf_trunc=0.01
    mesh_model.crop(pt1=(-0.15, -0.15, 0.01), pt2=(0.15, 0.15, 0.5))

    mesh_model.visualize()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh_model.mesh.vertices
    pcd.colors = mesh_model.mesh.vertex_colors
    
    # world coordinate
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    model_3d_noised.append(coordinate)
    model_3d_denoised.append(coordinate)

    # draw ray matching result
    line_colors = np.asarray([[1, 0, 0] for i in range(len(line_pts))])
    line_set = o3d.geometry.LineSet()
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    line_set.points = o3d.utility.Vector3dVector(line_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_ids)
    model_3d_noised.append(line_set)
    model_3d_denoised.append(line_set)

    # get candidate contact points candidate by find the nearest points of two lines
    print('[ Computing the candidate contact points ... ]')
    # get contact point candidate
    line_start_pts = [line_pts[2 * i] for i in range(len(line_pts) // 2)]
    p = nearest_point_of_all_lines(line_start_pts, line_vec)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20, create_uv_map=True)
    sphere.translate(p, relative=False)
    model_3d_noised.append(sphere)
    model_3d_denoised.append(sphere)

    # candidate_cp_pts = []
    # for i in range(len(line_vec)):
    #     for j in range(i + 1, len(line_vec)):
    #         if i != j:
    #             x1, x2 = nearset_point_of_two_lines(line_pts[i * 2], line_vec[i], line_pts[j * 2], line_vec[j])
    #             candidate_cp_pts.append(x1)
    #             candidate_cp_pts.append(x2)
    # candidate_cp_pts = np.array(candidate_cp_pts)
    # candidate_cp_pcd = o3d.geometry.PointCloud()
    # candidate_cp_pcd.points = o3d.utility.Vector3dVector(candidate_cp_pts)
    # candidate_cp_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(line_pts))])
    # model_3d_noised.append(candidate_cp_pcd)

    # runing DBSCAN of denoised contact points
    # print('[ Computing the clustered contact points ... ]')
    # clustered_cp_ind = DBSCAN_pcd(candidate_cp_pts, eps=0.005, min_samples=5, max_cluster=2)
    # if len(clustered_cp_ind) != 0: 
    #     clustered_cp_pts = candidate_cp_pts[clustered_cp_ind]
    #     clustered_cp_pcd = o3d.geometry.PointCloud()
    #     clustered_cp_pcd.points = o3d.utility.Vector3dVector(clustered_cp_pts)
    #     clustered_cp_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(clustered_cp_ind))])
    #     model_3d_denoised.append(clustered_cp_pcd)
    # else :
    #     print('--------------------------------------------')
    #     print('[ WARNING! the clustering result is empty! ]')
    #     print('--------------------------------------------')

    # add 3D point cloud (mesh)
    pcd_pts = np.array(pcd.points)
    pcd_colors = np.array(pcd.colors)
    model_3d_noised.append(pcd)
    
    # add 3D denoised point cloud (mesh)
    print('[ Computing the clustered point cloud ... ]')
    clustered_ind = DBSCAN_pcd(pcd_pts, eps=0.002, min_samples=10, max_cluster=1)
    clustered_pcd_pts = pcd_pts[clustered_ind]
    clustered_pcd_colors = pcd_colors[clustered_ind]

    clustered_pcd = o3d.geometry.PointCloud()
    if args.remove_lightbar:
        clustered_pcd.points = o3d.utility.Vector3dVector(clustered_pcd_pts)
        clustered_pcd.colors = o3d.utility.Vector3dVector(clustered_pcd_colors)

        # load light bar
        light_bar_pcd = o3d.io.read_point_cloud(f'{root}/light_bar/pcd.ply')
        light_bar_pts = np.asarray(light_bar_pcd.points)
        print(light_bar_pts.shape)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(clustered_pcd_pts)
        inds_remove = nn.kneighbors(light_bar_pts, return_distance=False)
        inds_mask = np.ones(len(clustered_pcd_pts), dtype=bool)
        inds_mask[inds_remove,] = False

        clustered_pcd = o3d.geometry.PointCloud()
        clustered_pcd.points = o3d.utility.Vector3dVector(clustered_pcd_pts[inds_mask])
        clustered_pcd.colors = o3d.utility.Vector3dVector(clustered_pcd_colors[inds_mask])
        print(f'[num of points before removing lignt bar : {len(clustered_pcd_pts)}]')
        print(f'[num of points after removing lignt bar : {len(clustered_pcd.points)}')

        model_3d_denoised.append(clustered_pcd)
    else :
        clustered_pcd.points = o3d.utility.Vector3dVector(clustered_pcd_pts)
        clustered_pcd.colors = o3d.utility.Vector3dVector(clustered_pcd_colors)
        model_3d_denoised.append(clustered_pcd)


    # save the result
    clustered_pcd.estimate_normals()
    distances = clustered_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radious = 0.5 * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(clustered_pcd, o3d.utility.DoubleVector([radious, radious * 2]))
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    
    # visualize all the results
    o3d.visualization.draw_geometries(model_3d_noised)
    o3d.visualization.draw_geometries(model_3d_denoised)
    o3d.visualization.draw_geometries([mesh])
    
    
    o3d.io.write_triangle_mesh("{}/mesh.obj".format(output_dir), mesh)
    o3d.io.write_point_cloud("{}/mesh.ply".format(output_dir), clustered_pcd)
    print("{}/mesh.obj saved".format(output_dir))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='cam_output', type=str)
    parser.add_argument('--in-dir', '-id', default='', type=str)
    parser.add_argument('--out-dir', '-od', default='annotation', type=str)
    parser.add_argument('--remove-lightbar', '-rlb', action='store_true')
    args = parser.parse_args()
    main(args)