import numpy as np
import cv2
import os
import glob
import scipy.io as sio

if __name__=="__main__":
    input_dir = 'rgbd/hcis-3cam'
    
    depth_dirs = glob.glob(f'{input_dir}/Depth*')
    color_dirs = glob.glob(f'{input_dir}/RGB*')

    print(depth_dirs)
    print(color_dirs)
    #  616.952; 617.113; 324.24; 239.484
    intrinsic = [
        np.array([
            [615.505,       0, 326.698],
            [      0, 615.511, 241.151],
            [      0,       0,       1],
        ], dtype=float), # 1
        np.array([
            [616.952,       0, 324.24],
            [      0, 617.113, 239.484],
            [      0,       0,       1],
        ], dtype=float), # 2
        np.array([
            [616.358,       0, 322.071],
            [      0, 616.988, 237.108],
            [      0,       0,       1],
        ], dtype=float) # 0
    ]

    depth_scale = [
        1000,
        1000,
        1000
    ]

    depth_list = []
    depth_plist = []
    for depth_dir in depth_dirs:
        depth_paths = glob.glob(f'{depth_dir}/*.npy')

        for depth_path in depth_paths:
            f = open(depth_path, 'rb')
            depth = np.load(f)
            depth_list.append(depth)
            depth_plist.append(depth_path)

    color_list = []
    color_plist = []
    for color_dir in color_dirs:
        color_paths = glob.glob(f'{color_dir}/*.png')

        for color_path in color_paths:
            color = cv2.imread(color_path)
            color_list.append(color)
            color_plist.append(color_path)
            
    length = len(depth_list)

    for i in range(length):
        camid = int(depth_plist[i].split('/')[-2][-1])
        save_dict = {
            'color': color_list[i],
            'depth': depth_list[i],
            'intr': intrinsic[camid],
            'dscale': depth_scale[camid]
        }
        path = os.path.splitext(depth_plist[i])[0] + '.mat'
        sio.savemat(path, save_dict)
        print(path, camid, ' saved')
