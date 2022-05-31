import os
import numpy as np
import cv2
import scipy.io as sio
from tqdm import tqdm

from utils.tag_detector import TagDetector


# default_dscale = 8000

# for ???
# default_intr = np.array(
#     [[622.423095, 0.0       , 313.393676],
#     [0.0       , 622.423217, 237.040466],
#     [0.0       , 0.0       , 1.0       ]])

# for D435
default_intr = np.array(
    [[617.10357666,         0.0,  328.6126709],
     [         0.0, 617.1975708, 234.39453125],
     [         0.0,         0.0,          1.0]])

def haskey(dict, key):
    return key in dict.keys()

def mat_loader(dpath, depth_scale, **kwargs):
    samples = []
    for sname in tqdm(os.listdir(dpath), desc="Loading samples"):
        spath = os.path.join(dpath, sname)
        smp = {
            'spath' : spath,
            # 'idx': int(sname[6:-4]) # ex : sample0.mat -> 0
        }
        data = sio.loadmat(spath)
        # C Is Contiguous layout. Mathematically speaking, row majo, other order : K(keep order), F(Fortran contiguous layout, column order), A(Is any order, Generally dont use this)
        smp['color'] = data['color'].astype(np.uint8, 'C')
        smp['depth'] = data['depth'].astype(np.uint16, 'C')
        smp['intr'] = data.get('intr', default_intr) # if 'intr' not exists, return default_intr
        smp['dscale'] = data.get('dscale', depth_scale)
        if not haskey(smp, 'pose'):
            tagDt = TagDetector(**kwargs)
            try:
                smp['pose'] = tagDt.esti_pose(smp['color'], smp['intr'])
            except Exception as e:
                print(e)
                print("Sample (", smp['spath'], ") is skipped!")
                continue
            del tagDt
        samples.append(smp)
    return samples

def png_loader(dpath, depth_scale, **kwargs):
    samples = []
    snames = [sname[:6] for sname in os.listdir(dpath) if sname.endswith('r.png')]
    for sname in tqdm(snames, desc="Loading samples"):
        spath = os.path.join(dpath, sname)
        smp = {
            'spath': spath,
            # 'idx': int(sname.split('-')[0])
        }
        smp['color'] = cv2.cvtColor(cv2.imread(spath + '-color.png'), cv2.COLOR_BGR2RGB)
        smp['depth'] = cv2.imread(spath + '-depth.png', cv2.IMREAD_ANYDEPTH).astype(np.uint16)
        if not haskey(smp, 'intr'): smp['intr'] = default_intr
        try:
            smp['pose'] = np.loadtxt(spath + '-pxose.txt')
        except IOError:
            tagDt = TagDetector(**kwargs)
            try:
                smp['pose'] = tagDt.esti_pose(smp['color'], smp.get('intr', default_intr))
            except Exception as e:
                print(e)
                print("Sample (", spath, ") is skipped!")
                continue
        if not haskey(smp, 'dscale'): smp['dscale'] = depth_scale
        samples.append(smp)
    return samples