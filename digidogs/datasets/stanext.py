"""
Dataset for Stanford Extra Dataset (Benjamin Bigs et al.)
"""
import os 
import json
import random
import torch
import kornia as K 
import numpy as np  
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl 
from pycocotools.mask import decode as decode_RLE
from digidogs.utils.normaliser import normalise_skeleton


class StanExt(Dataset):
    def __init__(self, 
            data_dir, 
            data_type='train', 
            transform=None, 
            out_res=64,
            sigma=1): 

       self.data_dir = data_dir 
       self.anno_dir = os.path.join(data_dir, 'animalpose_hg8_v0_results_on_StanExt')
       self.transform = transform 
       self.out_res = out_res
       self.sigma = sigma

       # === load the indices of the stanford extra dataset for the type of dataset ===  
       if data_type == "train": 
           self.indices = np.load(os.path.join(self.data_dir, "annotations/train_stanford_StanfordExtra_v12.npy")) # 6774 (54%) 
       elif data_type == "val": 
           self.indices = np.load(os.path.join(self.data_dir, "annotations/val_stanford_StanfordExtra_v12.npy")) # 4062 (32%) 
       elif data_type == "test": 
           self.indices = np.load(os.path.join(self.data_dir, "annotations/test_stanford_StanfordExtra_v12.npy")) # 1703 (13%) 
       # === load the dictionary containing data from the StanExt dataset === 
       jfile = open(os.path.join(self.data_dir, "annotations/StanfordExtra_v12.json"))
       self.jdata = np.array(json.load(jfile)) 
       jfile.close()
       # get rid of the multiple dogs... 
       temp_indices = []
       for i in self.indices: 
           if not self.jdata[i]["is_multiple_dogs"]:
               #print(np.asarray(self._extract_segmap(self.jdata[i])).max())
               # validation has no segmentation
               if np.asarray(self._extract_segmap(self.jdata[i])).sum() > 1:
                   # chec if file exists 
                   if os.path.exists(os.path.join(self.anno_dir, self.jdata[i]['img_path'].split('.')[0]+".json")): 
                       temp_indices.append(i)

       self.jdata = self.jdata[temp_indices]

    def __len__(self): 
        return len(self.jdata)

    def __getitem__(self, idx): 

        # === IMAGE EXTRACTION ===
        img_pth = os.path.join(self.data_dir, "data", self.jdata[idx]["img_path"])
        if not os.path.exists(img_pth): 
            img_pth = os.path.join(self.data_dir, "data", self.jdata[idx]["img_path"]).replace("jpg","png")
        image_pil = Image.open(img_pth).convert('RGB')
        image = np.array(image_pil)
        img_width = self.jdata[idx]["img_width"] 
        img_height = self.jdata[idx]["img_height"] 
        new_joint_pth = os.path.join(self.anno_dir, self.jdata[idx]['img_path'].split('.')[0]+".json")
        with open(new_joint_pth, 'r') as f: 
            anipose_data = json.load(f)
        # === THIS IS THE TARGET METADATA === 
        target = {}

        # === BBOX EXTRACTION === 
        boxx, boxy, boxw, boxh = self.jdata[idx]["img_bbox"] # current bounding box

        # set boundaries
        max_scale = min(img_width, img_height)
        #scale_crop = random.randint(0,img_width//4) 
        scale_crop=1
        minx = boxx - scale_crop
        miny = boxy - scale_crop
        maxx = (boxx+boxw) + scale_crop
        maxy = (boxy+boxh) + scale_crop
        box_width = maxx-minx
        box_height = maxy-miny
        side_length = max(box_width,box_height)
        x_center = (minx+maxx)//2
        y_center = (miny+maxy)//2
        xcropmin = x_center - side_length//2
        ycropmin = y_center - side_length//2
        xcropmax = xcropmin + side_length
        ycropmax = ycropmin + side_length
        crop_width = xcropmax - xcropmin
        crop_height = ycropmax - ycropmin

        # === KEYPOINT EXTRACTION === 
        joints = np.array(self.jdata[idx]["joints"]) 
        anipose_thr = 0.2
        anipose_joints_0to24 = np.asarray(anipose_data['anipose_joints_0to24']).reshape((-1, 3))
        anipose_joints_0to24_scores = anipose_joints_0to24[:, 2]
        anipose_joints_0to24_scores[anipose_joints_0to24_scores<anipose_thr] = 0.0
        anipose_joints_0to24[:, 2] = anipose_joints_0to24_scores
        #new_joints = np.concatenate((joints[:20, :], anipose_joints_0to24[20:24, :]), axis=0)
        anipose_joints_0to24[anipose_joints_0to24[:, 2]==0, :2] = 0     # avoid nan values
        x_temp = np.array(joints[:,0]) 
        y_temp = np.array(joints[:,1]) 
        v_temp = np.array(joints[:,2])
        k_indices = [-1,-1,-1,-1,-1,-1,2,-1,1,0,-1,8,-1,7,6,-1,5,4,3,-1,11,10,9,12,-1,13,16,17,14,15,18,19]
        assert len(k_indices) == 32, 'indices are not 32 keypoints'
        masked = k_indices != -1
        x = np.where(masked, x_temp[k_indices], -1)
        y = np.where(masked, y_temp[k_indices], -1)
        v = np.where(masked, v_temp[k_indices], -1)

        #x[2] = anipose_joints_0to24[22][0]
        #y[2] = anipose_joints_0to24[22][1]
        #v[2] = 1
        # === CROP IMAGE === 
        cropped_image = np.array(image_pil.crop((xcropmin, ycropmin, xcropmax, ycropmax)))
        #cropped_image = image[miny:maxy, minx:maxx]
        n_h, n_w, _ = cropped_image.shape

        # === SEGMENTATION EXTRACTION ===
        seg = self._extract_segmap(self.jdata[idx])
        crop_seg = Image.fromarray(seg*255) # originally it was bool the seg map
        crop_seg = np.array(crop_seg.crop((xcropmin, ycropmin, xcropmax, ycropmax)))

        #seg = seg[miny:maxy, minx:maxx]
        #bg = seg - 1

        # === TWEAK ORIGINAL COORDS TO LOOK LIKE THEY ARE IN THE 3D CAMERA SPACE === 
        focal_length = img_width * 1.2
        cx = (img_width /2) - xcropmin
        cy = (img_height /2) - ycropmin
        z3d = np.ones((len(x)))
        x3d = (x-xcropmin) - cx 
        y3d = (y-ycropmin) - cy 
        z3d = z3d * focal_length # we do not normalise it.. 
        x3d = torch.as_tensor(x3d)
        y3d = torch.as_tensor(y3d)
        z3d = torch.as_tensor(z3d)

        # === NORMALISE THOSE KEYPOINTS === 
        cam3d = torch.stack((x3d, y3d, z3d), dim=1)
        x3d, y3d, z3d = normalise_skeleton(cam3d, cx, cy, n_w, n_h, focal_length)
        xy_plane = torch.stack((x3d,y3d), dim=-1)
        zy_plane = torch.stack((z3d,y3d), dim=-1)
        xz_plane = torch.stack((x3d,z3d), dim=-1)

        # === HEATMAP EXTRACTION === 
        # we only multiply by 3 so it is uniformly with the 3d heatmaps
        heatmaps = torch.zeros((len(x)*3, self.out_res, self.out_res), dtype=torch.float32)

        for k_idx in range(len(x)): 
            if v[k_idx] > 0: 
                x0 = xy_plane[k_idx][0] 
                y0 = xy_plane[k_idx][1] 
                xy = torch.as_tensor([[x0,y0]])
                std = torch.ones_like(xy) * self.sigma
                heatmaps[k_idx*3+0]= K.geometry.subpix.render_gaussian2d(xy, std, (self.out_res, self.out_res), True)[0] 

                z = zy_plane[k_idx][0]
                y = zy_plane[k_idx][1]
                zy =torch.as_tensor([[z,y]])
                std = torch.ones_like(zy) * self.sigma
                heatmaps[k_idx*3+1]= K.geometry.subpix.render_gaussian2d(zy, std, (self.out_res, self.out_res), True)[0] # 2d heatmap

                x = xz_plane[k_idx][0]
                z = xz_plane[k_idx][1]
                xz =torch.as_tensor([[x,z]])
                std = torch.ones_like(xz) * self.sigma
                heatmaps[k_idx*3+2]= K.geometry.subpix.render_gaussian2d(xz, std, (self.out_res, self.out_res), True)[0] # 2d heatmap

        # === WE NEED TO TRANSFORM THE IMAGES ===
        if self.transform is not None: 
            image = self.transform(cropped_image)

        # === TRANSFORMS FOR THE SEGMENTATION MAPS === 
        crop_seg_t0 = torch.from_numpy(crop_seg) 
        crop_seg_t = F.resize(crop_seg_t0.unsqueeze(0), size=(448,448)) 
        seg_area = crop_seg_t.squeeze(0).numpy()
        #seg_area = torch.sqrt(torch.sum(seg_area==1))
        (sy, sx) = np.where(seg_area==255)
        smin_x, smin_y, smax_x, smax_y = sx.min(), sy.min(), sx.max(), sy.max()  
        s_w = smax_x - smin_x  
        s_h = smax_y - smin_y 
        seg_area = np.sqrt(s_w*s_h)
        crop_seg_t = F.resize(crop_seg_t0.unsqueeze(0), size=(self.out_res,self.out_res)) 
        final_seg = crop_seg_t.squeeze(0)
        #bg = seg_t(bg)

        # === TARGET === 
        final_3dkpts = [item for sublist in zip(x3d, y3d,z3d) for item in sublist] # 96
        #pts = [(x[k], y[k], v[k]) for k in range(len(x))]
        #target["keypoints3d"] = torch.as_tensor(np.array(final_3dkpts), dtype=torch.float32)
        target["keypoints3d"] = torch.as_tensor(np.array(final_3dkpts), dtype=torch.float32)
        target['cx'] = cx
        target['cy'] = cy
        target["hmps"] = heatmaps
        target['visiblity'] = torch.from_numpy(v).to(torch.float32) 
        target['nw'] = n_w
        target['nh'] = n_h
        target["seg"] = final_seg
        target['seg_area'] = seg_area
        #target['box'] = [smin_x, smin_y ,smax_x, smax_y]
        #target['label'] =self.jdata[idx]["img_path"].split('/')[0].split('-')[-1] 
        #target['img_pth'] = img_pth
        #if anipose_joints_0to24[22][2] <= 0:
        #if seg.sum()<1 : 
            #replacemend_index = max(0,idx-1)
            #image ,target = self.__getitem__(replacemend_index)
        return image, target 
    
    def _extract_segmap(self, entry): 
        """
        This code is from StanfordExtra repo.
        """
        rle = {
                "size":[entry['img_height'],entry['img_width']], 
                "counts": entry['seg']
                }
        decoded = decode_RLE(rle)
        return decoded 

