"""
Issue: memory leak of sampleManager
"""
#!/usr/bin/python3
import yaml
import cv2
import numpy as np
import open3d as o3d
import os
import glob
import matplotlib.pyplot as plt
from threading import Thread

from utils.mesh_model import MeshModel
from utils.loader import mat_loader, png_loader

__all__ = ["MeshModel", "SampleManager"]

def save_mesh_to_obj(mesh, path):

    with open(path, 'w') as f_out:
        vertices = np.asarray(mesh.vertices)  
        faces = np.asarray(mesh.triangles)

        # write vertices
        for v in vertices:
            f_out.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        
        # write faces
        for f in faces:
            f_out.write('f {} {} {}\n'.format(f[0], f[1], f[2]))

        f_out.close()

def estimate_center(sub_name):
    c_bbox = {}
    c_mesh = {}
    save_dir = "saved/"+sub_name
    for path in os.listdir(save_dir):
        if path.endswith('.ply'):
            mesh = MeshModel(os.path.join(save_dir,path)).mesh
            bbox = mesh.get_axis_aligned_bounding_box()
            key = path[:-4]
            c_mesh[key] = mesh.get_center().tolist()
            c_bbox[key] = bbox.get_center().tolist()
            # o3d.visualization.draw_geometries([mesh, bbox, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)])
    with open(save_dir+'/c_bbox.yaml', 'w') as file:
        yaml.safe_dump(c_bbox, file)
    with open(save_dir+'/c_mesh.yaml', 'w') as file:
        yaml.safe_dump(c_mesh, file)

class SampleManager:
    def __init__(self, sampleDir, depth_scale=8000, loader=mat_loader, **kwargs):
        self.kwargs = kwargs
        if isinstance(sampleDir, str):
            sampleDir = [sampleDir]
        # TODO multithread
        self.samples = []
        for dpath in sampleDir:
            print('\n',dpath)
            samples = loader(dpath, depth_scale, **kwargs)
            self.samples.extend(samples)
        self._meshModel = None
    
    @property
    def meshModel(self):
        if self._meshModel is None:
            self._meshModel = MeshModel()
            self._meshModel.tsdf_o3d([smp for smp in self.samples],
                                      voxel_length=0.001, sdf_trunc=0.003)
        return self._meshModel

    def show_sample_set(self):
        cv2.namedWindow('sample browser', cv2.WINDOW_AUTOSIZE)
        for smp in self.samples:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(smp['depth'], alpha=0.03), cv2.COLORMAP_JET)
            color_img = cv2.cvtColor(smp['color'], cv2.COLOR_RGB2BGR)
            images = np.hstack((color_img, depth_colormap))
            cv2.imshow(smp['spath'], images)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("img_save.png",images)
            elif key == ord('d'):
                cv2.destroyAllWindows()

    # def save_png_txt(self, dir, items=[]):
    #     for smp in self.samples:
    #         sence_id = smp['spath'].split('/')[-2]
    #         path_template = os.path.join(dir, sence_id, "%06d-{}.{}" %smp['idx'])
    #         os.makedirs(os.path.dirname(path_template),exist_ok=True)
    #         if 'color' in items:
    #             cv2.imwrite(path_template.format('color', 'png'), cv2.cvtColor(smp['color'], cv2.COLOR_RGB2BGR))
    #         if 'depth' in items:
    #             cv2.imwrite(path_template.format('depth', 'png'), smp['depth'].astype(np.uint16))
    #         if 'mask' in items and smp['mask'] is not None:
    #             cv2.imwrite(path_template.format('label', 'png'), smp['mask'])
    #         if 'pose' in items and smp['pose'] is not None:
    #             np.savetxt( path_template.format('pose' , 'txt'), smp['pose'])
    #     print(", ".join(items), 'saved')

    def _depth2pcd(sefl, depth, intrinsic, dscale, wordTcam, pt1=[-0.05,-0.05,0.003], pt2=[0.05,0.05,0.3]):
        width, height = depth.shape
        camMat = o3d.camera.PinholeCameraIntrinsic(
                width, height,
                intrinsic[0][0], intrinsic[1][1],
                intrinsic[0][2], intrinsic[1][2])
        depth = o3d.geometry.Image(depth)
        # create pcd
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camMat, wordTcam, depth_scale=dscale, project_valid_depth_only=True)
        bbox_pts = np.array([[pt1[0], pt1[1], pt1[2]],
                             [pt1[0], pt1[1], pt2[2]],
                             [pt1[0], pt2[1], pt1[2]],
                             [pt1[0], pt2[1], pt2[2]],
                             [pt2[0], pt1[1], pt1[2]],
                             [pt2[0], pt1[1], pt2[2]],
                             [pt2[0], pt2[1], pt1[2]],
                             [pt2[0], pt2[1], pt2[2]]])
        # crop ROI
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_pts))
        pcd = pcd.crop(bbox)

        return pcd

    def _xyz2uvz(self, xyz, intrinsic, height, width):
        uvz = np.dot(intrinsic, xyz)
        z = uvz[2]
        u = np.round(np.divide(uvz[0], z)).astype(np.int32)
        v = np.round(np.divide(uvz[1], z)).astype(np.int32)
        uvz_idx = np.where((u>=0) & (u<width) & (v>=0) & (v<height))
        return u[uvz_idx], v[uvz_idx], z[uvz_idx]

    # using fused model
    def approx_mask(self, border=True, registered_model=True, nr_dilate=2, nr_erode=3, nr_dilate_border=4):
        for smp in self.samples:
            if registered_model:# current not working
                model = self.meshModel.mesh
            else:
                model = self._depth2pcd(smp['depth'], smp['intr'], smp['dscale'], smp['pose'])

            model = model.__class__(model) # duplicate
            model.transform(smp['pose'])

            # reproject uv
            height, width = smp['depth'].shape
            xyz = np.asarray(model.vertices).T if registered_model else np.asarray(model.points).T
            u, v, _ = self._xyz2uvz(xyz, smp['intr'], height, width)

            # uv2mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[v, u] = 255 # Advanced Indexing

            # post process
            mask = cv2.dilate(mask, np.ones((3,3), dtype=np.uint8), iterations=nr_dilate)
            mask = cv2.erode(mask, np.ones((3,3), dtype=np.uint8), iterations=nr_erode)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
            # approx = cv2.approxPolyDP(contours[0], 2, True)
            approx = contours[0]

            mask = cv2.dilate(mask, np.ones((3,3), dtype=np.uint8), iterations=nr_dilate_border) if border else np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [approx] , 0, 1, -1)
            smp['mask'] = mask

        # o3d.visualization.draw_geometries([model, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])
        fig = plt.figure(num=smp['spath'], figsize=(8,6))
        ax = fig.add_subplot(1,2,1)
        ax.imshow(mask)
        ax = fig.add_subplot(1,2,2)
        display = smp['color']
        idx = np.where(mask == 255)
        display[idx[0], idx[1]] = 255
        ax.imshow(smp['color'])
        fig.tight_layout()
        plt.show()

    def __next__(self):
        try:
            smp = self.samples[self.curr_pos]
        except IndexError:
            raise StopIteration()
        self.curr_pos += 1
        return smp

    def __iter__(self):
        self.curr_pos = 0
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main(args):
    
    src_dir = os.path.join(args.root, args.in_dir)
    target_dir = os.path.join(args.root, args.out_dir)

    assert os.path.exists(args.root), f'{args.root} not exists'
    assert os.path.exists(src_dir), f'{args.root} not exists'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    try:

        sam_dirs = glob.glob(f'{src_dir}/*/mat')
        print(sam_dirs)
        
        smpMng = SampleManager(sam_dirs, depth_scale=0, loader=mat_loader, cx=0.092, cy=0.12, cz=0.003)
        mesh = smpMng.meshModel # voxel_length=0.002, sdf_trunc=0.01
        mesh.crop(pt1=(-0.25, -0.25, 0.002), pt2=(0.25, 0.25, 0.5))
        mesh.visualize()

        mesh.save("{}/mesh.ply".format(target_dir))
        print("{}/mesh.ply saved".format(target_dir))

        
    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default='cam_output', type=str)
    parser.add_argument('--in_dir', '-id', default='', type=str)
    parser.add_argument('--out_dir', '-od', default='', type=str)
    args = parser.parse_args()

    main(args)