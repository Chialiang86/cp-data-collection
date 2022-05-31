import cv2
from dt_apriltags import Detector
import math3d as m3d
import numpy as np
import argparse

"""
I. Detector Class (Pythonic wrapper for apriltag_detector.), see : https://github.com/duckietown/lib-dt-apriltags/blob/c04858205977c4e71dfd275cb74a6a0fb8178cb0/dt_apriltags/apriltags.py#L190

Detector(families='tag36h11',
        nthreads=6,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0).detect(rgb[:,:,2], False, intr, self.tsize)

1. families: Tag families, separated with a space, default: tag36h11
2. nthreads: Number of threads, default: 1
3. quad_decimate: Detection of quads can be done on a lower-resolution image, improving speed at a cost of pose accuracy and a slight 
   decrease in detection rate. Decoding the binary payload is still done at full resolution, default: 2.0
4. quad_sigma: What Gaussian blur should be applied to the segmented image (used for quad detection?)  Parameter is the standard deviation 
   in pixels.  Very noisy images benefit from non-zero values (e.g. 0.8), default:  0.0
5. refine_edges: When non-zero, the edges of the each quad are adjusted to "snap to" strong gradients nearby. This is useful when decimation 
   is employed, as it can increase the quality of the initial quad estimate substantially. Generally recommended to be on (1). Very 
   computationally inexpensive. Option is ignored if quad_decimate = 1, default: 1
6. decode_sharpening: How much sharpening should be done to decoded images? This can help decode small tags but may or may not help in odd 
   lighting conditions or low light conditions, default = 0.25
7. searchpath: Where to look for the Apriltag 3 library, must be a list, default: ['apriltags']
8. debug: If 1, will save debug images. Runs very slow, default: 0

II. math3d.Transform Class

def __init__(
        self, 
        orientation: Quat = Quat.identity(), 
        position: Vec3 = Vec3.zero()
    ):
        self.rotation = orientation
        self.translation = position


"""

class TagDetector:
    def __init__(self, nx=8, ny=12, cx=0.096, cy=0.12, cz=0, tsize=0.02, tspace=0.004):
        # Detect Apriltags
        self.nx, self.ny = nx, ny
        self.cx, self.cy, self.cz = cx, cy, cz
        self.tsize, self.tspace = tsize, tspace

    def get_obj_img_points(self, tags):
        objPoints = []# N*3
        imgPoints = []# N*2
        for tag in tags:
            tag_id = tag.tag_id
            x_off = (tag_id %self.nx) * (self.tsize+self.tspace)
            y_off = (tag_id//self.nx) * (self.tsize+self.tspace)
            objPoints.append([x_off              -self.cx, y_off              -self.cy, -self.cz])
            objPoints.append([x_off+self.tsize   -self.cx, y_off              -self.cy, -self.cz])
            objPoints.append([x_off+self.tsize   -self.cx, y_off+self.tsize   -self.cy, -self.cz])
            objPoints.append([x_off              -self.cx, y_off+self.tsize   -self.cy, -self.cz])
            objPoints.append([x_off+self.tsize/2 -self.cx, y_off+self.tsize/2 -self.cy, -self.cz])
            
            imgPoints.append(tag.corners[0])
            imgPoints.append(tag.corners[1])
            imgPoints.append(tag.corners[2])
            imgPoints.append(tag.corners[3])
            imgPoints.append(tag.center)
        return np.array(objPoints, dtype=np.float32), \
            np.array(imgPoints, dtype=np.float32)
    
    def tag_visualize(self, rgb, intr_mat):
        intr = (intr_mat[0,0], intr_mat[1,1], intr_mat[0,2], intr_mat[1,2])
        tags = Detector(families='tag36h11',
                               nthreads=6,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0).detect(rgb[:,:,2], False, intr, self.tsize)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(bgr, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
            cv2.putText(bgr, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
        cv2.imshow('tags', bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # TODO: return data on stack(this will lead to error)
    def esti_pose(self, rgb, intr_mat):
        intr = (intr_mat[0,0], intr_mat[1,1], intr_mat[0,2], intr_mat[1,2])
        tags = Detector(families='tag36h11',
                               nthreads=6,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0).detect(rgb[:,:,2], False, intr, self.tsize)
        objpts, imgpts = self.get_obj_img_points(tags)
        if len(objpts) < 15:
            raise Exception("The number of detected tag points is not enough to estimate the imgae pose!")
        pose =  cv2.solvePnP(objpts, imgpts, intr_mat, None)[1:] # retval,rvec,tvec
        pose_mat = m3d.Transform( np.vstack((pose[1], pose[0])).flatten() ).array
        return pose_mat

def main(args):
    import scipy.io as sio
    tagDect = TagDetector()

    data = sio.loadmat('../dataset/{}/sample0.mat'.format(args.obj))
    print(tagDect.esti_pose(data['color'], data['intr']))
    tagDect.tag_visualize(data['color'], data['intr'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', '-o', default='cup_1', type=str)
    args = parser.parse_args()

    main(args)