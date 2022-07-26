import numpy as np
import argparse
import os
import scipy.io as sio
import cv2
import threading

from utils.realsense_cam import realsense_cam

def capture(cam, intr, mat_prefix, rgb_prefix):

    rgb, depth = cam.get_image()

    mat_out = {
        'color': rgb,
        'depth': depth,
        'intr' : intr,
        'dscale' : 1 / cam.depth_scale
    }

    mat_path = f'{mat_prefix}.mat'
    rgb_path = f'{rgb_prefix}.png'

    cv2.imwrite(rgb_path, rgb)
    sio.savemat(mat_path, mat_out)
    print(f'[{mat_path} , {rgb_path}] saved')

def main(args):
    serial_nums = realsense_cam.get_realsense_serial_num()
    print('serial nums', serial_nums)

    root = args.root
    out_dir = args.out_dir

    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, out_dir), exist_ok=True)

    cams = []
    cam_dirs = []
    cam_intrs = []
    for i, serial_num in enumerate(serial_nums):
        cam = realsense_cam(serial_num=serial_num)
        cams.append(cam)

        cam_dir = os.path.join(root, out_dir, f'{serial_num}')
        os.makedirs(cam_dir, exist_ok=True)
        cam_dirs.append(cam_dir)

        intr = cam.intrinsic_mat
        cam_intrs.append(intr)

        intr_path = os.path.join(cam_dir, 'intrinsic.npy')
        cam.write_intrinsics(intr_path)
    

    for i in range(len(cams)):
        cv2.namedWindow(f'image_{i} : serial_num = {serial_nums[i]}', cv2.WINDOW_AUTOSIZE)

    rgbs = [None] * len(cams) 
    depths = [None] * len(cams) 
    # index = [0] * len(cams)
    idx = 0

    first = True
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

            # create directory if needed 
            for cam_dir in cam_dirs:
                if not os.path.exists(os.path.join(cam_dir, 'rgb_static')):
                    os.mkdir(os.path.join(cam_dir, 'rgb_static'))
                if not os.path.exists(os.path.join(cam_dir, 'mat_static')):
                    os.mkdir(os.path.join(cam_dir, 'mat_static'))

            worker = []
            for i in range(len(cams)):
                mat_prefix = cam_dirs[i] + '/mat_static/' + str(idx)
                rgb_prefix = cam_dirs[i] + '/rgb_static/' + str(idx)
                worker.append(threading.Thread(target=capture, args=(cams[i], cam_intrs[i], mat_prefix, rgb_prefix)))
            
            for i in range(len(worker)):
                worker[i].start()

            for i in range(len(worker)):
                worker[i].join()
            
            idx += 1

        if key == ord('a'):

            # create directory if needed 
            for cam_dir in cam_dirs:
                if not os.path.exists(os.path.join(cam_dir, 'rgb_dynamic')):
                    os.mkdir(os.path.join(cam_dir, 'rgb_dynamic'))
                if not os.path.exists(os.path.join(cam_dir, 'mat_dynamic')):
                    os.mkdir(os.path.join(cam_dir, 'mat_dynamic'))

            for index in range(100):
                _ = cv2.waitKey(100)
                worker = []
                for i in range(len(cams)):
                    mat_prefix = cam_dirs[i] + '/mat_dynamic/' + str(index)
                    rgb_prefix = cam_dirs[i] + '/rgb_dynamic/' + str(index)
                    worker.append(threading.Thread(target=capture, args=(cams[i], cam_intrs[i], mat_prefix, rgb_prefix)))
                
                for i in range(len(worker)):
                    worker[i].start()

                for i in range(len(worker)):
                    worker[i].join()
            

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
    