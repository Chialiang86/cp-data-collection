import open3d as o3d
import numpy as np
import os
import cv2

def main():
    obj_path = '3d_scene/20220729_183526_scissor/dynamic_pcd_0.ply'

    pcd = o3d.io.read_point_cloud(obj_path)

    o3d.visualization.draw_geometries([pcd])
    

if __name__=="__main__":
    main()