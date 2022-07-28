import argparse
import scipy.io as sio
import os 
import glob
import numpy as np
import json
import cv2


def main(args):
    input_root = args.input_root
    input_dir = os.path.join(args.input_root, args.input_dir)
    output_root = os.path.join(args.output_root, args.input_dir)

    assert os.path.exists(input_root) and os.path.exists(input_dir), f'input dir not exists'
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    # config output dir
    output_cam_dirs = os.listdir(input_dir)
    output_cam_dirs.sort()
    for output_cam_dir in output_cam_dirs:
        os.makedirs(os.path.join(output_root, output_cam_dir), exist_ok=True)
        output_static_dir = os.path.join(output_root, output_cam_dir, 'mat_static')
        output_dynamic_dir = os.path.join(output_root, output_cam_dir, 'mat_dynamic')
        os.makedirs(output_static_dir, exist_ok=True)
        os.makedirs(output_dynamic_dir, exist_ok=True)
    
    input_cam_dirs = os.listdir(input_dir)
    input_cam_dirs.sort()

    for idx, input_cam_dir in enumerate(input_cam_dirs):
        print(f'processing {input_dir}/{input_cam_dir} ...')
        intr_path = f'{input_dir}/{input_cam_dir}/intrinsic.npy'
        color_paths = glob.glob(f'{input_dir}/{input_cam_dir}/*_*.jpg')
        depth_paths = glob.glob(f'{input_dir}/{input_cam_dir}/*_*.npy')
        color_paths.sort(key=lambda s: int(os.path.splitext(s)[0].split('_')[-1]))
        depth_paths.sort(key=lambda s: int(os.path.splitext(s)[0].split('_')[-1]))

        output_static_dir = os.path.join(output_root, output_cam_dirs[idx], 'mat_static')
        output_dynamic_dir = os.path.join(output_root, output_cam_dirs[idx], 'mat_dynamic')
        assert len(color_paths) == len(depth_paths)
        for i in range(len(color_paths)):

            rgb = cv2.imread(color_paths[i])
            depth = np.load(depth_paths[i])
            intr = np.load(intr_path)
            mat_out = {
                'color': rgb,
                'depth': depth,
                'intr' : intr,
                'dscale' : 1000 # d435 default
            }

            if i == 0:
                output_path = os.path.join(output_static_dir, f'0.mat')
            else :
                output_path = os.path.join(output_dynamic_dir, f'{i}.mat')
            
            sio.savemat(output_path, mat_out)
            print(f'{output_path} saved')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', '-ir', type=str, default='hcis-data')
    parser.add_argument('--input-dir', '-id', type=str, default='20220727_172155_scissor')
    parser.add_argument('--output-root', '-or', type=str, default='cam_output')
    args = parser.parse_args()
    main(args)