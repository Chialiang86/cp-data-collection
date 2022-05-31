import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

class MeshModel:
    def __init__(self, path=None):
        self.mesh = None
        if not path is None:
            self.mesh = o3d.io.read_triangle_mesh(path)

    def tsdf_o3d(self, smpMng, voxel_length=0.0015, sdf_trunc=0.003):
        if not hasattr(self, 'tsdfVolume'):
            print("",
                "Create TSDF Volume:",
                "    voxel_length: %.3f" %voxel_length,
                "    sdf_trunc:    %.3f" %sdf_trunc,
                sep="\n")
            self.tsdfVolume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_length,
                sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        for smp in tqdm(smpMng, desc="Processing Images"):
            intr_mat = smp['intr']
            heigh, width = smp['depth'].shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, heigh, 
                    intr_mat[0,0], intr_mat[1,1], intr_mat[0,2], intr_mat[1,2])
            color, depth = o3d.geometry.Image(smp['color']), o3d.geometry.Image(smp['depth'])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=smp['dscale'], convert_rgb_to_intensity=False)
            self.tsdfVolume.integrate(rgbd, intrinsic, smp['pose'])
        self.mesh = self.tsdfVolume.extract_triangle_mesh()
        
    def crop(self, pt1, pt2):
        bbox_pts = np.array([[pt1[0], pt1[1], pt1[2]],
                             [pt1[0], pt1[1], pt2[2]],
                             [pt1[0], pt2[1], pt1[2]],
                             [pt1[0], pt2[1], pt2[2]],
                             [pt2[0], pt1[1], pt1[2]],
                             [pt2[0], pt1[1], pt2[2]],
                             [pt2[0], pt2[1], pt1[2]],
                             [pt2[0], pt2[1], pt2[2]]])
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bbox_pts)
        )
        self.mesh = self.mesh.crop(bbox)

    def transform(self, tranMat):
        self.mesh = self.mesh.transform(tranMat)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        o3d.io.write_triangle_mesh(path, self.mesh, write_vertex_colors=True)

    def visualize(self, color_uniform=None, coordinate_size=0.1):
        model_list = []
        if not color_uniform is None:
            mesh = o3d.geometry.TriangleMesh(self.mesh.vertices, self.mesh.triangles)
            mesh.paint_uniform_color(color_uniform)
        else:
            mesh = self.mesh
        model_list.append(mesh)

        if coordinate_size > 0:
            model_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_size, origin=[0, 0, 0]))

        o3d.visualization.draw_geometries(model_list)
