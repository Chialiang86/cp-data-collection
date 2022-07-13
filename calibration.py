import numpy as np
from utils.realsense_cam import realsense_cam
import os, cv2, glob, time, joblib, json, threading

# for multi cam
from utils.utils import solve_pose, print_m3d_pose
from utils.minisam_optim import opt_poses
from utils.tag_detection import tag_boards, detect_tags, get_tagboard_obj_pts

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

def multicam_calib(cam1, cam2, board_name='board4x6', target_path='./calibration/multicam/'):
    os.makedirs(target_path, exist_ok=True)

    cam1Tcam2_seq, i = [], 1

    # tagboard_dict shape : [nx * ny, 5, 3]
    # tag_size of board4x6 : 0.04 (4cm)
    tagboard_dict, tag_size = tag_boards(board_name)
    while True:
        # get img
        color_img1, _ = cam1.get_image()
        color_img2, _ = cam2.get_image()

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
            print('Pose {} saved'.format(i))
            i += 1
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
        'extr_opt':  optim_cam1Tcam2.get_matrix().tolist()
    }

    with open(f'{target_path}{cam1.serial_num}-{cam2.serial_num}.json', 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
        print(f'{target_path}{cam1.serial_num}-{cam2.serial_num}.json saved.')

    # joblib.dump(cam1Tcam2_seq, f'{target_path}{cam1.serial_num}-{cam2.serial_num}_seq.pkl')
    # joblib.dump(optim_cam1Tcam2, f'{target_path}optim_{cam1.serial_num}-{cam2.serial_num}.pkl')
    # print(f'{target_path}{cam1.serial_num}-{cam2.serial_num}_seq.pkl saved.')
    # print(f'{target_path}optim_{cam1.serial_num}-{cam2.serial_num}.pkl saved.')


def multicam_calib(cam, board_name='board4x6', target_path='./calibration/camTtag/'):
    os.makedirs(target_path, exist_ok=True)

    camTtag_seq, i = [], 1

    # tagboard_dict shape : [nx * ny, 5, 3]
    # tag_size of board4x6 : 0.04 (4cm)
    tagboard_dict, tag_size = tag_boards(board_name)
    while True:
        # get img
        color_img, _ = cam.get_image()

        # detect tags
        detect_img, tag_IDs, tag_img_pts = detect_tags(color_img, cam.intrinsic_at, tag_size)

        return_char = cv2.waitKey(50)
        if return_char & 0xFF == ord('s') and len(tag_IDs) > 10 :
            tag_img_pts = np.array(tag_img_pts).reshape(-1, 2)
            tag_obj_pts = get_tagboard_obj_pts(tagboard_dict, tag_IDs)

            m3d_transform = solve_pose(tag_obj_pts, tag_img_pts, cam.intrinsic_mat)
            camTtag_seq.append(m3d_transform)
            print('Pose {} saved'.format(i))
            i += 1
        elif return_char & 0xFF == 27:  # esc
            cv2.destroyAllWindows()
            break

    optim_camTtag = opt_poses(camTtag_seq) # use minisam
    
    json_dict = {
        'master': cam.serial_num,
        'extr_seq': [cam1Tcam2.get_matrix().tolist() for cam1Tcam2 in camTtag_seq],
        'extr_opt':  optim_camTtag.get_matrix().tolist()
    }

    with open(f'{target_path}{cam.serial_num}-tag.json', 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
        print(f'{target_path}{cam.serial_num}-tag.json saved.')

    # joblib.dump(cam1Tcam2_seq, f'{target_path}{cam1.serial_num}-{cam2.serial_num}_seq.pkl')
    # joblib.dump(optim_cam1Tcam2, f'{target_path}optim_{cam1.serial_num}-{cam2.serial_num}.pkl')
    # print(f'{target_path}{cam1.serial_num}-{cam2.serial_num}_seq.pkl saved.')
    # print(f'{target_path}optim_{cam1.serial_num}-{cam2.serial_num}.pkl saved.')

def main():
    serial_nums = realsense_cam.get_realsense_serial_num()
    serial_nums.sort() # smallest one as master

    cams = []
    for serial_num in serial_nums:
        cam = realsense_cam(serial_num=serial_num)
        cams.append(cam)

    print('''
    ==== please input key to continue ====

        i : for intrinsic calibration
        e : for extrinsic calibration
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

            # for i in range(len(workers)):
            #     workers[i].start()

            # for i in range(len(workers)):
            #     workers[i].join()
        
        if key == 'e': # extrinsic
            if len(cams) >= 2:
                # workers = []
                for i in range(1, len(cams)):
                    # workers.append(threading.Thread(target=multicam_calib, args=(cams[0], cams[i + 1], 'board4x6', './calibration/multicam/')))
                    multicam_calib(cams[0], cams[i], board_name='board4x6', target_path='./calibration/multicam/')

                # for i in range(len(workers)):
                #     workers[i].start()

                # for i in range(len(workers)):
                #     workers[i].join()
    
if __name__=="__main__":
    main()
