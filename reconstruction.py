import argparse
import numpy as np
import scipy.io as sio
import glob
import os
import open3d as o3d
import json


def tsdf_fusion(rgbds, voxel_length=0.0015, sdf_trunc=0.003):
    
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for rgbd_info in rgbds:
        height, width = rgbd_info['depth'].shape
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


def main(args):
    root = args.root
    in_dir = os.path.join(root, args.in_dir)
    extr_dir = args.extr_dir

    assert os.path.exists(root), f'{root} not exists'
    assert os.path.exists(in_dir), f'{in_dir} not exists'
    assert os.path.exists(extr_dir), f'{extr_dir} not exists'

    mat_files = glob.glob(f'{in_dir}/*/mat_static/*')
    extr_files = glob.glob(f'{extr_dir}/*.json')

    assert len(mat_files) > 0, 'no mat files given'
    assert len(extr_files) > 0, 'no extrinsic given'

    # master_id = extr_files[0].split('/')[-1].split('-')[0] # ex : calibration/multicam/745212070452-740112071376.json'

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
        rgbd_info['dscale'] = mat['dscale']
        rgbd_info['intr'] = mat['intr']

        for extrinsic_dict in extrinsic_dicts:
            if extrinsic_dict['slave'] == cam_id:
                rgbd_info['extr'] = np.asarray(extrinsic_dict['extr_opt'])
                break 
            elif extrinsic_dict['master'] == cam_id:
                rgbd_info['extr'] = np.identity(4)
                break 

        rgbds.append(rgbd_info)

    tsdf_fusion(rgbds)
        
    
        




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='cam_output', type=str)
    parser.add_argument('--in-dir', '-id', default='', type=str)
    parser.add_argument('--extr-dir', '-ed', default='calibration/multicam', type=str)
    parser.add_argument('--out-dir', '-od', default='3D_model', type=str)
    args = parser.parse_args()
    main(args)