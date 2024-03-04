import os 
import torch
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import kornia as K 
import cv2
import glob
import numpy as np
from PIL import Image
import json
import random
from digidogs.rgbddog import utils
from digidogs.utils.normaliser import normalise_skeleton,denormalise_skeleton
torch.set_printoptions(threshold=torch.inf)
np.set_printoptions(threshold=np.inf)

class RgbdDog(Dataset):
    def __init__(self, data_dir, train_type='train', transform=None, test_dog=None): 
        self.transform = transform
        self.sigma=0.005
        self.out_res = 32
        if train_type == 'train':
            train_pth = os.path.join(data_dir, 'rgbddog_json', 'train_rgbd.json')
            with open(train_pth, 'r') as pfile: 
                self.jdata = json.load(pfile)
        if train_type == 'test' and test_dog is not None:
            test_pth = os.path.join(data_dir, 'rgbddog_json', f'test{test_dog}_rgbd.json')
            with open(test_pth, 'r') as pfile: 
                self.jdata = json.load(pfile)

    def __len__(self):
        return len(self.jdata['images'])

    def __getitem__(self,idx): 

        # --- calib ---
        intrincsics = self.jdata['calibs'][idx]['K']
        fx = intrincsics[0][0]
        fy = intrincsics[1][1]
        cx = intrincsics[0][2]
        cy = intrincsics[1][2]
        # --- joints ---
        bvhjoints = np.array(self.jdata['joints'][idx]) # camera joints
        # --- mask --- 
        mask = cv2.imread(self.jdata['masks'][idx], -1)
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST )
        pil_image =Image.open(self.jdata['images'][idx]).convert('RGB') 
        image = np.array(pil_image)
        img_width = image.shape[1]
        img_height = image.shape[0]
        #mask = mask.astype(np.uint8)
        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices)!=0 : 
            assert len(y_indices) == len(x_indices), "indices do not have the same length"
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            ymin = np.min(y_indices)
            ymax = np.max(y_indices)
            scale_ = min(img_width,img_height)
            scale_crop = random.randint(1,scale_//4) 
            xmin = xmin - scale_crop
            ymin = ymin - scale_crop
            xmax = xmax + scale_crop 
            ymax = ymax + scale_crop
            box_width = xmax-xmin
            box_height = ymax-ymin
            side_length = max(box_width,box_height)
            x_center = (xmin+xmax)//2
            y_center = (ymin+ymax)//2
            xcropmin = x_center - side_length//2
            ycropmin = y_center - side_length//2
            xcropmax = xcropmin + side_length
            ycropmax = ycropmin + side_length
            crop_width = xcropmax - xcropmin
            crop_height = ycropmax - ycropmin

            image = np.array(pil_image.crop((xcropmin, ycropmin, xcropmax, ycropmax)))
            mask_pil = Image.fromarray(mask)
            mask_crop  = mask_pil.crop((xcropmin, ycropmin, xcropmax, ycropmax))
            mask = np.array(mask_crop)
            #image = image[ycropmin:ycropmax,xcropmin:xcropmax]
            #mask = np.array(mask[ycropmin:ycropmax,xcropmin:xcropmax])

            # how to add, make it square
            n_h, n_w = image.shape[:2]
            k_indices=[0,1,2,18,19,3,4,5,6,7,9,10,11,12,13,25,26,27,28,30,31,32,33,35,38,42,20,-1,21,23,22,24] 
            assert len(k_indices) == 32, 'indices are not 32 keypoints'
            masked = k_indices != -1
            x = torch.from_numpy(np.where(masked, bvhjoints[:,0][k_indices], -1))
            y = torch.from_numpy(np.where(masked, bvhjoints[:,1][k_indices], -1))
            z = torch.from_numpy(np.where(masked, bvhjoints[:,2][k_indices], -1))
            pts3d = torch.stack((x,y,z),dim=1).reshape(32,3) 

            # --- normalise ---
            x3d, y3d, z3d = normalise_skeleton(pts3d, cx-xcropmin*2, cy-ycropmin*2, n_w, n_h, fx, fy=fy, scale=0.5)
            n_pts3d = torch.stack((x3d,y3d,z3d),dim=1).reshape(32,3) 
            new_x, new_y, new_z = denormalise_skeleton(n_pts3d, cx-xcropmin*2, cy-ycropmin*2, n_w, n_h)
            new_pts3d = torch.stack((new_x,new_y,new_z),dim=1).reshape(32,3) 
            norm_length = np.linalg.norm(pts3d[3] - pts3d[0])
            norm_lengthv1 = np.linalg.norm(new_pts3d[3] - new_pts3d[0])
            #print(norm_length, norm_lengthv1)

            xy_plane = torch.stack((x3d,y3d), dim=-1)
            zy_plane = torch.stack((z3d,y3d), dim=-1)
            xz_plane = torch.stack((x3d,z3d), dim=-1)
            
            ## === GENERATE GROUND TRUTH ===
            v = torch.ones(32)
            v[-5] = -1
            target = torch.zeros(len(x)*3, self.out_res, self.out_res, dtype=torch.float32)
            for k_idx in range(len(x)): 
                if v[k_idx] > 0: # if keypoint is visible
                    x0 = xy_plane[k_idx][0]
                    y0 = xy_plane[k_idx][1]
                    xy = torch.as_tensor([[x0,y0]])
                    std = torch.ones_like(xy) * self.sigma
                    target[k_idx*3+0]= K.geometry.subpix.render_gaussian2d(xy, std, (self.out_res, self.out_res), True)[0] # 2d heatmap

                    z = zy_plane[k_idx][0]
                    y = zy_plane[k_idx][1]
                    zy =torch.as_tensor([[z,y]])
                    std = torch.ones_like(zy) * self.sigma
                    target[k_idx*3+1]= K.geometry.subpix.render_gaussian2d(zy, std, (self.out_res, self.out_res), True)[0] # 2d heatmap

                    x = xz_plane[k_idx][0]
                    z = xz_plane[k_idx][1]
                    xz =torch.as_tensor([[x,z]])
                    std = torch.ones_like(xz) * self.sigma
                    target[k_idx*3+2]= K.geometry.subpix.render_gaussian2d(xz, std, (self.out_res, self.out_res), True)[0] # 2d heatmap

            if self.transform is not None: 
                image = self.transform(image)
            seg = torch.from_numpy(mask)
            seg = seg/seg.max()
            crop_seg_t = F.resize(seg.unsqueeze(0), size=(448,448)) 
            seg_area = crop_seg_t.squeeze(0)
            #print(seg_area)
            #seg_area = torch.sum(seg_area==255)
            (sy, sx) = np.where(seg_area > 0) 
            smin_x, smin_y = sx.min(), sy.min() 
            smax_x, smax_y = sx.max(), sy.max()  
            s_w = smax_x - smin_x  
            s_h = smax_y - smin_y 
            seg_area = np.sqrt(s_w*s_h)
            crop_seg_t = F.resize(seg.unsqueeze(0), size=(self.out_res,self.out_res)) 
            final_seg = crop_seg_t.squeeze(0)

            final_3dkpts = [item for sublist in zip(x3d, y3d, z3d) for item in sublist] # 96
            target = {
                    'keypoints3d':torch.FloatTensor(final_3dkpts).reshape(-1,3), 
                    'cx': cx - xcropmin, 
                    'cy': cy - ycropmin, 
                    'hmps': target.to(torch.float32),
                    'visiblity' : v.to(torch.float32), 
                    'nw': n_w,
                    'nh': n_h, 
                    'seg': final_seg,
                    'seg_area':seg_area,
                    'norm_3d': norm_length
                    }

        else:
            replacemend_index = max(0,idx-1)
            image ,target = self.__getitem__(replacemend_index)

        return image, target

