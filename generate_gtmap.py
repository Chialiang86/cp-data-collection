import cv2
import os
import glob
import numpy as np

# rgb(232,187,66)

def weighted_image(img, weight=[-3,1,2]):
    float_img = np.array(img).astype(float)
    ret = np.zeros(img.shape[:2])
    for i, w in enumerate(weight):
        ret += w * float_img[:,:,i] ** 2
    ret = ((ret - np.min(ret)) / (np.max(ret) - np.min(ret)) * 255).astype(np.uint8)

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

def main():
    files = glob.glob(f'img/*.jpg')
    for file in files:
        # if '_haetmap.png' in file or '_weighted.png' in file:
        #     continue
       
        prefix = os.path.splitext(file)[0]
        # write to path
        path_heatmap = f'{prefix}_haetmap.png'
        path_weighted = f'{prefix}_weighted.png'
        # if os.path.exists(path_heatmap) and os.path.exists(path_weighted):
        #     continue
        
        print(f'processing {file} ...')
        img = cv2.imread(file)
        weighted_render = weighted_image(img)
        
        (condx, condy) = np.where(weighted_render == np.max(weighted_render))
        max_xy = np.vstack((condx, condy)).T

        # for xy in max_xy:
        #     cv2.circle(img, (xy[1], xy[0]), 10, (255, 0, 0), 2)

        conf = heatmap(img.shape[:2], max_xy, 150)
        conf_render = cv2.addWeighted(img, 0.3, conf, 0.7, 0)

        cv2.imwrite(path_heatmap, conf_render)
        cv2.imwrite(path_weighted, weighted_render)

        # print(f'file : {file}, max \n : {condx} {condy}')

if __name__=="__main__":
    main()