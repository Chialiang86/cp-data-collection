import open3d as o3d
import numpy as np
import os
import cv2

def main():
    mesh_path = '3d_scene/20220727_180937_Lego/mesh.obj'
    obj_path = '3d_scene/20220727_180937_Lego/pcd.ply'

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = o3d.io.read_point_cloud(obj_path)

    o3d.visualization.draw_geometries([mesh])
    o3d.visualization.draw_geometries([pcd])
    

if __name__=="__main__":
    main()