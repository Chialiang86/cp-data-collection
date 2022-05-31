import cv2
import numpy as np
import math3d as m3d


def print_m3d_pose(pose: m3d.Transform, name='Current pose'):
    print(f'{name}: rx, ry, rz, x, y, z')
    print(np.degrees(pose.orient.to_euler('XYZ')), pose.pos.array)


def wait_esc():
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # esc
            break


def solve_pose(obj_pts, img_pts, cam_mat, dist_coeffs=np.zeros(4), flags=cv2.SOLVEPNP_IPPE) -> m3d.Transform:
    """
    :param img_pts: image points in 2D, np.array (n, 2)
    :param obj_pts: physical points in 3D, np.array (n, 3)
    :param cam_mat: camera intrinsics, np.array (3, 3)
    :param flags: cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE
    :return: 4x4 m3d.Transform
    """
    # camera see the object
    retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mat, dist_coeffs, flags=flags)
    rot_mat, _ = cv2.Rodrigues(rvec)
    camTobj = np.vstack((np.hstack((rot_mat, tvec.reshape(-1, 1))), np.array([0, 0, 0, 1])))
    
    return m3d.Transform(camTobj)


def unit_transform(value, type='N2g'):
    if type == 'N2g':
        return value / 9.80665 * 1e-3
    elif type == 'g2N':
        return value / 1e-3 * 9.80665
    else:
        print(f'No such \'{type}\' unit transform.')
        return -1
