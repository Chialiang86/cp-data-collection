import numpy as np
import math3d as m3d
import transforms3d as tf3d
import argparse
from utils.realsense_cam import realsense_cam
import os, cv2, glob, time, joblib, json, threading
from scipy.spatial.transform import Rotation as R

# for multi cam
from utils.utils import solve_pose, print_m3d_pose
from utils.minisam_optim import opt_poses
from utils.tag_detection import tag_boards, detect_tags, get_tagboard_obj_pts

from handeye_calib.dual_quaternion import DualQuaternion as dq
from handeye_calib.dual_quaternion_hand_eye_calibration import HandEyeConfig, \
    compute_hand_eye_calibration_RANSAC, compute_hand_eye_calibration_BASELINE
import handeye_calib.transformations as trf
# from ur10_robot import UR10Robot

def cam_intr_calibration(cam : realsense_cam, img_size : tuple=(640, 480), board_size : tuple=(8, 6), target_root : str='./calibration'):
    target_path = f'{target_root}/{cam.serial_num}/'
    os.makedirs(target_path, exist_ok=True)

    # take imgs of the chessboard
    cam.enable_InfraRed(False)
    i = 0
    while True:
        color_img, _ = cam.get_image()
        cv2.imshow(f'Current View : {cam.serial_num}', color_img)
        return_char = cv2.waitKey(10) & 0xFF
        if return_char == 27: # ESC
            break
        elif return_char == ord('s'):
            cv2.imwrite(f'{target_path}{i}.jpg', color_img)
            print(f'{target_path}{i}.jpg saved.')
            i += 1
    
    if i == 0:
        cv2.destroyAllWindows()
        return

    # camera calibration
    corner_x, corner_y = board_size[0] - 1, board_size[1] - 1
    objp = np.zeros((corner_x * corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)
    obj_pts = []  # 3d points in real world space
    img_pts = []  # 2d points in image plane.
    imgs = glob.glob(f'{target_path}*.jpg')

    for fname in imgs:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        print('find the chessboard corners of', fname)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        # If found, add object points, image points
        if ret == True:
            obj_pts.append(objp)
            img_pts.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)
            cv2.imshow(f'Current View : {cam.serial_num}', img)
            _ = cv2.waitKey(500)

    print('Camera calibration...')
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)

    if rms > 0.7:
        print('Poor quality, please do the calibration again')
        return 

    json_dict = {
        'serial_num': cam.serial_num,
        'intr_before': cam.intrinsic_mat.tolist(),
        'intr_after':  camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }

    with open(f'{target_path}intrinsics.json', 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
        print(f'{target_path}intrinsics.json saved.')

    # joblib.dump((camera_matrix, dist_coeffs), f'{target_path}intrinsics.pkl')
    # print(f'{target_path}intrinsics.pkl saved.')

    cv2.destroyAllWindows()

def multicam_calib_from_png(src_dir, board_name='board4x6', target_path='./calibration/multicam/'):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    
    cam1_serialnum = src_dir.split('/')[-1].split('-')[0]
    cam2_serialnum = src_dir.split('/')[-1].split('-')[1]

    # get intrinsic
    cam1_intr_f = f'{src_dir}/{cam1_serialnum}.npy'
    cam1_intr = np.load(cam1_intr_f) # (fx, fy, ox, oy)
    cam1_intr_at = (cam1_intr[0, 0], cam1_intr[1, 1], cam1_intr[0, 2], cam1_intr[1, 2])
    cam2_intr_f = f'{src_dir}/{cam2_serialnum}.npy'
    cam2_intr = np.load(cam2_intr_f) # (fx, fy, ox, oy)
    cam2_intr_at = (cam2_intr[0, 0], cam2_intr[1, 1], cam2_intr[0, 2], cam2_intr[1, 2])
    
    # get rgb images
    cam1_imgs = glob.glob(f'{src_dir}/{cam1_serialnum}*.png')
    cam2_imgs = glob.glob(f'{src_dir}/{cam2_serialnum}*.png')
    assert len(cam1_imgs) == len(cam2_imgs), f'different file numbers in {src_dir}/{cam1_serialnum} and {src_dir}/{cam2_serialnum}'

    cam1_imgs.sort(key=lambda s: int(s.split('-')[-1].split('.')[0])) # ex : xxx/141722071222-0.png
    cam2_imgs.sort(key=lambda s: int(s.split('-')[-1].split('.')[0])) # ex : xxx/141722071222-0.png

    cam1Tcam2_seq = []
    tagboard_dict, tag_size = tag_boards(board_name)
    for i in range(len(cam1_imgs)):

        color_img1 = cv2.imread(cam1_imgs[i])
        color_img2 = cv2.imread(cam2_imgs[i])
        
        # detect tags
        detect_img1, tag_IDs1, tag_img_pts1 = detect_tags(color_img1, cam1_intr_at, tag_size)
        detect_img2, tag_IDs2, tag_img_pts2 = detect_tags(color_img2, cam2_intr_at, tag_size)
        detect_imgs = np.hstack([detect_img1, detect_img2])
        cv2.imshow('multicam_calib', detect_imgs)
        cv2.waitKey(50)

        if len(tag_IDs1) > 10 and len(tag_IDs2) > 10:
            tag_img_pts1 = np.array(tag_img_pts1).reshape(-1, 2)
            tag_img_pts2 = np.array(tag_img_pts2).reshape(-1, 2)
            tag_obj_pts1 = get_tagboard_obj_pts(tagboard_dict, tag_IDs1)
            tag_obj_pts2 = get_tagboard_obj_pts(tagboard_dict, tag_IDs2)

            m3d_transform1 = solve_pose(tag_obj_pts1, tag_img_pts1, cam1_intr)
            m3d_transform2 = solve_pose(tag_obj_pts2, tag_img_pts2, cam2_intr)
            cam1Tcam2 = m3d_transform1 * m3d_transform2.inverse
            print_m3d_pose(cam1Tcam2)
            cam1Tcam2_seq.append(cam1Tcam2)
    cv2.destroyAllWindows()
    
    # optim_cam1Tcam2 = opt_poses(cam1Tcam2_seq) # use minisam
    # print_m3d_pose(optim_cam1Tcam2, f'Optimal {cam1_serialnum} to {cam2_serialnum} pose')

    pos_eulers = []
    for cam1Tcam2 in cam1Tcam2_seq:
        cam1Tcam2_np = cam1Tcam2.get_matrix()
        pos_euler = np.concatenate((cam1Tcam2_np[:3, 3], R.from_matrix(cam1Tcam2_np[:3, :3]).as_rotvec()))
        pos_eulers.append(pos_euler)
    pos_eulers = np.asarray(pos_eulers)
    pos_euler_mean = np.mean(pos_eulers, axis=0)
    optim_cam1Tcam2 = np.identity(4)
    optim_cam1Tcam2[:3, :3] = R.from_rotvec(pos_euler_mean[3:]).as_matrix()
    optim_cam1Tcam2[:3, 3] = pos_euler_mean[:3]
    
    json_dict = {
        'master': cam1_serialnum,
        'slave': cam2_serialnum,
        # 'extr_seq': [cam1Tcam2.get_matrix().tolist() for cam1Tcam2 in cam1Tcam2_seq],
        # 'extr_opt':  optim_cam1Tcam2.get_matrix().tolist()
        'extr_opt':  optim_cam1Tcam2.tolist()
    }

    with open(f'{target_path}{cam1_serialnum}-{cam2_serialnum}.json', 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
        print(f'{target_path}{cam1_serialnum}-{cam2_serialnum}.json saved.')


def multicam_calib(cam1, cam2, board_name='board4x6', target_path='./calibration/multicam/', save_rgb=True):
    os.makedirs(target_path, exist_ok=True)
    if save_rgb:
        save_rgb_path = os.path.join(target_path, f'{cam1.serial_num}-{cam2.serial_num}')
        os.makedirs(save_rgb_path, exist_ok=True)

    cnt = 0
    cam1Tcam2_seq = []
    optim_cam1Tcam2 = np.zeros((4, 4))

    # tagboard_dict shape : [nx * ny, 5, 3]
    # tag_size of board4x6 : 0.04 (4cm)
    tagboard_dict, tag_size = tag_boards(board_name)
    while True and cnt < 20:
        # get img
        color_img1, _ = cam1.get_image()
        color_img2, _ = cam2.get_image()

        if save_rgb:
            cv2.imwrite(f'{save_rgb_path}/{cam1.serial_num}-{cnt}.png', color_img1)
            cv2.imwrite(f'{save_rgb_path}/{cam2.serial_num}-{cnt}.png', color_img2)

        # detect tags
        detect_img1, tag_IDs1, tag_img_pts1 = detect_tags(color_img1, cam1.intrinsic_at, tag_size)
        detect_img2, tag_IDs2, tag_img_pts2 = detect_tags(color_img2, cam2.intrinsic_at, tag_size)
        detect_imgs = np.hstack([detect_img1, detect_img2])
        cv2.imshow('multicam_calib', detect_imgs)

        return_char = cv2.waitKey(50)
        if return_char & 0xFF == ord('s') and len(tag_IDs1) > 10 and len(tag_IDs2) > 10:
            tag_img_pts1 = np.array(tag_img_pts1).reshape(-1, 2)
            tag_img_pts2 = np.array(tag_img_pts2).reshape(-1, 2)
            tag_obj_pts1 = get_tagboard_obj_pts(tagboard_dict, tag_IDs1)
            tag_obj_pts2 = get_tagboard_obj_pts(tagboard_dict, tag_IDs2)

            m3d_transform1 = solve_pose(tag_obj_pts1, tag_img_pts1, cam1.intrinsic_mat)
            m3d_transform2 = solve_pose(tag_obj_pts2, tag_img_pts2, cam2.intrinsic_mat)
            cam1Tcam2 = m3d_transform1 * m3d_transform2.inverse
            print_m3d_pose(cam1Tcam2)
            cam1Tcam2_seq.append(cam1Tcam2)
            cnt += 1
            print('Pose {} saved'.format(cnt))
        elif return_char & 0xFF == 27:  # esc
            # cam1.process_end()
            # cam2.process_end()
            cv2.destroyAllWindows()
            break
    
    optim_cam1Tcam2 = opt_poses(cam1Tcam2_seq) # use minisam
    print_m3d_pose(optim_cam1Tcam2, f'Optimal {cam1.serial_num} to {cam2.serial_num} pose')

    
    json_dict = {
        'master': cam1.serial_num,
        'slave': cam2.serial_num,
        'extr_seq': [cam1Tcam2.get_matrix().tolist() for cam1Tcam2 in cam1Tcam2_seq],
        'extr_opt':  optim_cam1Tcam2.tolist()
    }

    with open(f'{target_path}{cam1.serial_num}-{cam2.serial_num}.json', 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
        print(f'{target_path}{cam1.serial_num}-{cam2.serial_num}.json saved.')

    # joblib.dump(cam1Tcam2_seq, f'{target_path}{cam1.serial_num}-{cam2.serial_num}_seq.pkl')
    # joblib.dump(optim_cam1Tcam2, f'{target_path}optim_{cam1.serial_num}-{cam2.serial_num}.pkl')
    # print(f'{target_path}{cam1.serial_num}-{cam2.serial_num}_seq.pkl saved.')
    # print(f'{target_path}optim_{cam1.serial_num}-{cam2.serial_num}.pkl saved.')


# Pose form: x, y, z, q_x, q_y, q_z, q_w
def solve_relative_transformation(base_c1, base_c2):
    # Usage 1. base_T_hand,world_T_eye (eye on hand) ->Return: handTeye[baseline/RANSAC]
    # Usage 2. world_T_cam1,world_T_cam2 -> cam1Tcam2 ->Return: cam1Tcam2[baseline/RANSAC]
    conf = HandEyeConfig()
    conf.min_num_inliers = len(base_c1) // 1.2
    conf.prefilter_poses_enabled = False
    conf.enable_exhaustive_search = True
    pose_base_hand = [dq.from_pose_vector(p) for p in base_c1]  # Hand in robot-base frame
    pose_world_eye = [dq.from_pose_vector(p) for p in base_c2]  # Eye in world frame
    result_baseline = compute_hand_eye_calibration_BASELINE(pose_base_hand, pose_world_eye, conf)
    # result_RANSAC = compute_hand_eye_calibration_RANSAC(pose_base_hand, pose_world_eye, conf)
    quat_bl = result_baseline[1].to_pose()
    # quat_rs = result_RANSAC[1].to_pose()
    # x, y, z, q_x, q_y, q_z, q_w
    c1Tc2_bl = trf.quaternion_matrix([quat_bl[6], quat_bl[3], quat_bl[4], quat_bl[5]])  # w,x,y,z
    c1Tc2_bl[0:3, 3] = quat_bl[0:3]
    # c1Tc2_rs = trf.quaternion_matrix([quat_rs[6], quat_rs[3], quat_rs[4], quat_rs[5]])  # w,x,y,z
    # c1Tc2_rs[0:3, 3] = quat_rs[0:3]
    return c1Tc2_bl  # , c1Tc2_rs


# def handeye_calib_ur10(cam, board_name='board4x6',
#                        target_path='./calibration/handeye_calib_ur10/',
#                        handTeye_name='handTeye', baseThand_name='baseThand', worldTeye_name='worldTeye'):
#     # baseThand: ur10Ttcp
#     # worldTeye: tagboardTcam2
#     # handTeye: tcpTcam2

#     # baseThand: ur10Ttcp
#     # worldTeye: cam3Tgrip
#     # handTeye: tcpTgrip

#     os.makedirs(target_path, exist_ok=True)

#     robot = UR10Robot()
#     origin_ur10Ttcp_seq = []
#     baseThand_seq, worldTeye_seq = [], []
#     i = len(baseThand_seq) + 1
#     tagboard_dict, tag_size = tag_boards(board_name)

#     while True:
#         color_img, _ = cam.get_image()
#         detect_img, tag_IDs, tag_img_pts = detect_tags(color_img, cam.intrinsic_at, tag_size)
#         tag_obj_pts = get_tagboard_obj_pts(tagboard_dict, tag_IDs)
#         cv2.imshow('handeye_calib_ur10', detect_img)

#         return_char = cv2.waitKey(50)
#         if return_char & 0xFF == ord('s') and len(tag_IDs) > 10:
#             ur10Ttcp_i = robot.get_pose()
#             origin_ur10Ttcp_seq.append(ur10Ttcp_i)
#             quad_bTh = tf3d.quaternions.mat2quat(ur10Ttcp_i.orient.array)  # qw,qx,qy,qz
#             quad_bTh = quad_bTh[[1, 2, 3, 0]].tolist()  # qx,qy,qz,qw
#             quad_bTh = ur10Ttcp_i.pos.array.tolist() + quad_bTh
#             baseThand_seq.append(np.array(quad_bTh))

#             worldTeye_i = solve_pose(tag_obj_pts, np.array(tag_img_pts).reshape(-1, 2), cam.intrinsic_mat).inverse
#             quad_wTcam = tf3d.quaternions.mat2quat(worldTeye_i.orient.array)
#             quad_wTcam = quad_wTcam[[1, 2, 3, 0]].tolist()  # qx,qy,qz,qw
#             quad_wTcam = worldTeye_i.pos.array.tolist() + quad_wTcam
#             worldTeye_seq.append(np.array(quad_wTcam))

#             print('Pose {} saved'.format(i))
#             joblib.dump(origin_ur10Ttcp_seq, f'{target_path}origin_ur10Ttcp_seq.pkl')
#             joblib.dump(baseThand_seq, f'{target_path}{baseThand_name}_seq.pkl')
#             joblib.dump(worldTeye_seq, f'{target_path}{worldTeye_name}_seq.pkl')
#             print(f'{target_path}origin_ur10Ttcp_seq.pkl saved.')
#             print(f'{target_path}{baseThand_name}_seq.pkl saved.')
#             print(f'{target_path}{worldTeye_name}_seq.pkl saved.')
#             i += 1
#         elif return_char & 0xFF == 27:  # esc
#             cam.process_end()
#             robot.close()
#             cv2.destroyAllWindows()
#             break

#     handTeye = solve_relative_transformation(np.vstack(baseThand_seq), np.vstack(worldTeye_seq))
#     optim_handTeye = m3d.Transform(handTeye)
#     print_m3d_pose(optim_handTeye, f'Optimal {handTeye_name} pose')

#     # dump pose (for future use)
#     joblib.dump(optim_handTeye, f'{target_path}optim_{handTeye_name}.pkl')
#     joblib.dump(baseThand_seq, f'{target_path}{baseThand_name}_seq.pkl')
#     joblib.dump(worldTeye_seq, f'{target_path}{worldTeye_name}_seq.pkl')
#     print(f'{target_path}optim_{handTeye_name}.pkl saved.')
#     print(f'{target_path}{baseThand_name}_seq.pkl saved.')
#     print(f'{target_path}{worldTeye_name}_seq.pkl saved.')


def solve_transform(target_path='./calibration/handeye_calib_ur10/',
                    handTeye_name='handTeye', baseThand_name='baseThand', worldTeye_name='worldTeye'):
    baseThand_seq = joblib.load(f'{target_path}{baseThand_name}_seq.pkl')
    worldTeye_seq = joblib.load(f'{target_path}{worldTeye_name}_seq.pkl')

    handTeye = solve_relative_transformation(np.vstack(baseThand_seq), np.vstack(worldTeye_seq))
    optim_handTeye = m3d.Transform(handTeye)
    print_m3d_pose(optim_handTeye, f'Optimal {handTeye_name} pose')

    # dump pose (for future use)
    joblib.dump(optim_handTeye, f'{target_path}optim_{handTeye_name}.pkl')
    joblib.dump(baseThand_seq, f'{target_path}{baseThand_name}_seq.pkl')
    joblib.dump(worldTeye_seq, f'{target_path}{worldTeye_name}_seq.pkl')
    print(f'{target_path}optim_{handTeye_name}.pkl saved.')
    print(f'{target_path}{baseThand_name}_seq.pkl saved.')
    print(f'{target_path}{worldTeye_name}_seq.pkl saved.')

def main(args):

    ext_dir = args.ext_dir

    serial_nums = realsense_cam.get_realsense_serial_num()
    serial_nums.sort() # smallest one as master

    cams = []
    for serial_num in serial_nums:
        cam = realsense_cam(serial_num=serial_num)
        cams.append(cam)

    print('''
    ==== please input key to continue ====

        i : for intrinsic calibration
        e : for extrinsic calibration from scratch
        es : for extrinsic calibration from saved rgb
        q : break

    ======================================
          ''')
    
    while True:

        key = str(input('> '))
        if key == 'q': 
            for cam in cams:
                cam.process_end()
            break

        if key == 'i': # intrinsic
            # workers = []
            for i in range(len(cams)):
                # cv2.namedWindow(f'Current View : {cam.serial_num}')
                # workers.append(threading.Thread(target=cam_intr_calibration, args=(cams[i], (640, 480), (8, 6))))
                cam_intr_calibration(cams[i], img_size=(640, 480), board_size=(8, 6))
        
        if key == 'e': # extrinsic
            if len(cams) >= 2:
                # workers = []
                target_path = f'./calibration/{ext_dir}/'
                assert os.path.exists(target_path), f'{target_path} not exists'
                for i in range(1, len(cams)):
                    # workers.append(threading.Thread(target=multicam_calib, args=(cams[0], cams[i + 1], 'board4x6', './calibration/multicam/')))
                    multicam_calib(cams[0], cams[i], board_name='board4x6', target_path=f'./calibration/{target_path}/')

        if key == 'es': # extrinsic

            root = f'calibration/{ext_dir}'
            assert os.path.exists(root), f'{root} not exists'
            dirs = os.listdir(root)

            for dir in dirs:
                src_dir = os.path.join(root, dir)
                if os.path.isdir(src_dir):
                    src_dir = os.path.join(root, dir)
                    multicam_calib_from_png(src_dir, board_name='board4x6')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext_dir', '-ed', type=str, default='multicam')
    args = parser.parse_args()
    main(args)
