import argparse
import numpy as np
import scipy.io as sio
import glob
import os
import open3d as o3d
from tqdm import tqdm
import json

def create_rgbd(rgb, depth, intr, extr, dscale):
    assert rgb.shape[:2] == depth.shape
    (h, w) = depth.shape
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    ix, iy =  np.meshgrid(range(w), range(h))

    x_ratio = (ix.ravel() - cx) / fx
    y_ratio = (iy.ravel() - cy) / fy

    z = depth.ravel() / dscale
    x = z * x_ratio
    y = z * y_ratio

    points = np.vstack((x, y, z)).T
    colors = np.reshape(rgb,(depth.shape[0] * depth.shape[1], 3))
    colors = np.array([colors[:,2], colors[:,1], colors[:,0]]).T / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(extr)

    return pcd

def scene_reconstruction(rgbds):

    pcds = []
    for rgbd_info in rgbds:

        pcd = create_rgbd(rgbd_info['color'], rgbd_info['depth'], rgbd_info['intr'], rgbd_info['extr'], rgbd_info['dscale'])
        pcds.append(pcd)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    pcds.append(coordinate)
    o3d.visualization.draw_geometries(pcds)

def crop_mesh(mesh, pt1, pt2):
    bbox_pts = np.array([[pt1[0], pt1[1], pt1[2]],
                            [pt1[0], pt1[1], pt2[2]],
                            [pt1[0], pt2[1], pt1[2]],
                            [pt1[0], pt2[1], pt2[2]],
                            [pt2[0], pt1[1], pt1[2]],
                            [pt2[0], pt1[1], pt2[2]],
                            [pt2[0], pt2[1], pt1[2]],
                            [pt2[0], pt2[1], pt2[2]]])
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bbox_pts)
    )
    mesh = mesh.crop(bbox)

def tsdf_fusion(rgbds, voxel_length=0.005, sdf_trunc=0.015):
    
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for rgbd_info in tqdm(rgbds):
        height, width = rgbd_info['depth'].shape
        rgbd_info['color'] = rgbd_info['color'][...,::-1].copy()
        color, depth = o3d.geometry.Image(rgbd_info['color']), o3d.geometry.Image(rgbd_info['depth'])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, 
            height,
            rgbd_info['intr'][0, 0],
            rgbd_info['intr'][1, 1],
            rgbd_info['intr'][0, 2],
            rgbd_info['intr'][1, 2],
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=rgbd_info['dscale'], convert_rgb_to_intensity=False
                )
        tsdf_volume.integrate(rgbd, intrinsic, np.linalg.inv(rgbd_info['extr']))
    
    mesh = tsdf_volume.extract_triangle_mesh()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([coordinate, mesh])
    o3d.visualization.draw_geometries([coordinate, pcd])

    return mesh, pcd

def reconstruct_scene(extr_files, mat_files, out_dir):
    # extract master
    extrinsic_dicts = []
    for extr_file in extr_files:
        with open(extr_file, 'r') as f:
            j_dict = json.load(f)
        extrinsic_dicts.append(j_dict)
    
    rgbds = []
    for mat_file in mat_files:
        cam_id = mat_file.split('/')[-3] # ex : cam_output/good/745212070452/mat_static/0.mat
        mat = sio.loadmat(mat_file)

        rgbd_info = {}
        rgbd_info['color'] = mat['color'].astype(np.uint8, 'C')
        rgbd_info['depth'] = mat['depth'].astype(np.uint16, 'C')
        rgbd_info['intr'] = mat['intr']
        rgbd_info['dscale'] = mat['dscale']

        for extrinsic_dict in extrinsic_dicts:
            if extrinsic_dict['slave'] == cam_id:
                rgbd_info['extr'] = np.asarray(extrinsic_dict['extr_opt'])
                break 
            elif extrinsic_dict['master'] == cam_id:
                rgbd_info['extr'] = np.identity(4)
                break 
        
        rgbds.append(rgbd_info)

    # scene_reconstruction(rgbds)

    mesh, pcd = tsdf_fusion(rgbds)
    o3d.io.write_triangle_mesh(f'{out_dir}/mesh.obj', mesh)
    o3d.io.write_point_cloud(f'{out_dir}/pcd.ply', pcd)

def main(args):
    root = args.root
    in_dir = os.path.join(root, args.in_dir)
    out_dir = os.path.join('3d_scene', args.out_dir)
    extr_dir = args.extr_dir

    assert os.path.exists(root), f'{root} not exists'
    assert os.path.exists(in_dir), f'{in_dir} not exists'
    assert os.path.exists(extr_dir), f'{extr_dir} not exists'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # master_id = extr_files[0].split('/')[-1].split('-')[0] # ex : calibration/multicam/745212070452-740112071376.json'

    # for static scene
    static_mat_files = glob.glob(f'{in_dir}/*/mat_static/*')
    static_extr_files = glob.glob(f'{extr_dir}/*.json')

    assert len(static_mat_files) > 0, 'no mat files given'
    assert len(static_extr_files) > 0, 'no extrinsic given'
    reconstruct_scene(static_extr_files, static_mat_files, out_dir)

    # # for dynamic scene
    # dynamic_mat_dirs = glob.glob(f'{in_dir}/*')
    # print(dynamic_mat_dirs)
    # dynamic_mat_files = glob.glob(f'{in_dir}/*/mat_dynamic/*')
    # dynamic_extr_files = glob.glob(f'{extr_dir}/*.json')

    # assert len(dynamic_mat_files) > 0, 'no mat files given'
    # assert len(dynamic_extr_files) > 0, 'no extrinsic given'
    # static_scene_pcd = reconstruct_scene(dynamic_extr_files, dynamic_mat_files)
    # o3d.io.write_point_cloud(f'{out_dir}/static.ply', static_scene_pcd)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='cam_output', type=str)
    parser.add_argument('--in-dir', '-id', default='', type=str)
    parser.add_argument('--out-dir', '-od', default='', type=str)
    parser.add_argument('--extr-dir', '-ed', default='calibration/multicam', type=str)
    args = parser.parse_args()
    main(args)