import enum
import numpy as np
import argparse
import os
import scipy.io as sio
import cv2

from utils.realsense_cam import realsense_cam

def main(args):
    serial_nums = realsense_cam.get_realsense_serial_num()
    print('serial nums', serial_nums)

    root = args.root
    out_dir = args.out_dir

    if not os.path.exists(root):
        os.mkdir(root)
    assert not os.path.exists(os.path.join(root, out_dir)), f'{os.path.join(root, out_dir)} exists'
    os.mkdir(os.path.join(root, out_dir))

    cams = []
    cam_dirs = []
    cam_intrs = []
    for i, serial_num in enumerate(serial_nums):
        cam = realsense_cam(serial_num=serial_num)
        cams.append(cam)

        cam_dir = os.path.join(root, out_dir, f'{serial_num}')
        os.mkdir(cam_dir)
        os.mkdir(os.path.join(cam_dir, 'rgb'))
        os.mkdir(os.path.join(cam_dir, 'mat'))
        cam_dirs.append(cam_dir)

        intr = cam.intrinsic_mat
        cam_intrs.append(intr)

        intr_path = os.path.join(cam_dir, 'intrinsic.npy')
        cam.write_intrinsics(intr_path)
    

    for i in range(len(cams)):
        cv2.namedWindow(f'image_{i} : serial_num = {serial_nums[i]}', cv2.WINDOW_AUTOSIZE)

    rgbs = [None] * len(cams) 
    depths = [None] * len(cams) 
    index = [0] * len(cams)
    while True:
        key = cv2.waitKey(10)

        for i in range(len(cams)):
            rgb, depth = cams[i].get_image()
            rgbs[i] = rgb
            depths[i] = depth
            depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            viewer_frams = np.hstack((rgb, depth))
            cv2.imshow(f'image_{i} : serial_num = {serial_nums[i]}', viewer_frams)

        if key == ord('s'):
            for i in range(len(cams)):
                rgb_out = rgbs[i]
                mat_out = {
                    'color': rgb_out,
                    'depth': depths[i],
                    'intr' : cam_intrs[i],
                    'dscale' : 1 / cams[i].depth_scale
                }

                mat_path = os.path.join(cam_dirs[i], 'mat', f'{index[i]}.mat')
                rgb_path = os.path.join(cam_dirs[i], 'rgb', f'{index[i]}.png')

                cv2.imwrite(rgb_path, rgb_out)
                sio.savemat(mat_path, mat_out)
                print(f'[{mat_path} , {rgb_path}] saved')
                index[i] += 1

        if key == ord('q'):
            cv2.destroyAllWindows()
            for cam in cams:
                cam.stop()
            break
                
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, default='cam_output', help='the output root directory')
    parser.add_argument('--out_dir', '-od', type=str, default='', help='the name of collecting directory')
    args = parser.parse_args()

    main(args)
    