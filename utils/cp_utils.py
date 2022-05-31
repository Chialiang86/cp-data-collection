from sklearn.cluster import DBSCAN
import numpy as np
import cv2

def weighted_image(img, weight=[2,1,-3]):
    float_img = np.array(img).astype(float)
    ret = np.zeros(img.shape[:2])
    for i, w in enumerate(weight):
        ret += w * float_img[:,:,i] ** 2
    ret = ((ret - np.min(ret)) / (np.max(ret) - np.min(ret)) * 255).astype(np.uint8)

    return ret

def weighted_pcd_colors(colors : np.ndarray,  weight=[2,1,-3]):
    ret = np.zeros(colors.shape[0])
    for i, w in enumerate(weight):
        ret += w * colors[:,i] ** 2
    ret = (ret - np.min(ret)) / (np.max(ret) - np.min(ret))
    
    return ret

def heatmap(imsize, xys, var):

    # (condx, condy) = np.where(weighted_render == np.max(weighted_render))
    # xys = np.vstack((condx, condy)).T

    assert len(imsize) == 2, 'imsize error'
    ret = np.zeros(imsize)
    gridr, gridc = np.meshgrid(range(imsize[1]), range(imsize[0]))
    for xy in xys:
        # gaussian distribution without non-exponent trem
        ret += np.exp(-(((gridc - xy[0])**2) + (gridr - xy[1])**2) / (2 * var)) 
    
    ret = ret / np.max(ret) * 255 # normalized
    ret = cv2.applyColorMap(ret.astype(np.uint8), cv2.COLORMAP_JET)
    return ret

def ray_marching_by_pose(pose, intrinsic, max_xy, scale):
    assert pose.shape == (4, 4), f'the pose shape is invalid : {pose.shape}'
    
    # project point
    xy = max_xy[0]

    # intrinsic param
    focal_len = (intrinsic[0, 0] + intrinsic[1, 1]) / 2
    center_r = intrinsic[0, 2]
    center_c = intrinsic[1, 2]
    cp_r = xy[1] - center_r 
    cp_c = xy[0] - center_c 

    unit_dir = np.array([[cp_r],[cp_c],[focal_len]])
    unit_dir /= np.sum(unit_dir) 
    unit_dir *= scale
    unit_dir = np.vstack((unit_dir, [[1]]))

    start = np.array([[0],[0],[0],[1]])
    start = pose @ start
    end = pose @ unit_dir

    return start[:3,-1].reshape(3), end[:3,-1].reshape(3)

def nearset_point_of_two_lines(r1, v1, r2, v2):
    v1 /= (np.sum(v1**2) ** 0.5) # normalized
    v2 /= (np.sum(v2**2) ** 0.5) # normalized
    v3 = np.cross(v1, v2) # perpendicular to v1, v2

    # r1 + t1 x v1 + t3 x v3 = r2 + t2 * v2 
    # t1 x v1 - t2 * v2 + t3 * v3 = r2 - r1
    # [ v1 -v2 v3 ] @ t = [ r2 - r1 ] 
    A = np.hstack((v1.reshape(3, -1), -v2.reshape(3, -1), v3.reshape(3, -1)))
    b = (r2 - r1).reshape(3, -1)
    t = np.linalg.inv(A) @ b #[ [t1], [t2], [t3]]

    nearest_r1 = r1 + t[0, 0] * v1
    nearest_r2 = r2 + t[1, 0] * v2

    return nearest_r1, nearest_r2

def DBSCAN_pcd(candidate_pts : np.ndarray, eps=0.001, min_samples=10, max_cluster=1):
    assert max_cluster > 0, f'max_cluster must greater than zero'
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(candidate_pts)
    clustering_labels = clustering.labels_
    clustering_ind = []
    # clustering_pts = []
    # clustering_colors = []

    cluster_sizes = []
    for i in range(np.max(clustering_labels)):
        cond = np.where(clustering_labels == i)
        cluster_sizes.append(-len(cond[0]))
    cluster_sizes = np.array(cluster_sizes)

    # sorting by clusters' size
    descending_order = np.argsort(cluster_sizes)

    # assign
    cnt = 0
    for i in descending_order:
        cnt += 1
        if cnt > max_cluster:
            break

        cond = np.where(clustering_labels == i)
        clustering_ind.extend(cond[0])
        pts = candidate_pts[cond]
        # clustering_pts.extend(pts)
        # clustering_colors.extend([[0, 1, 1 / np.max(clustering_labels) * i] for _ in range(len(cond[0]))])
    
    return np.array(clustering_ind) #, np.array(clustering_pts), np.array(clustering_colors)