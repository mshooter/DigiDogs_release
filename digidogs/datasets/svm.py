import os
import re
import glob
import numpy as np
import json
import torch
import kornia as K 
import torchvision.transforms as transforms
from torch.utils.data import Dataset 
from PIL import Image
from digidogs.utils.normaliser import normalise_skeleton
class Svm(Dataset):
    def __init__(self, data_dir="/vol/research/animo/datasets_internal/dogs_svm_mars_2019/", transform=None, out_res=32, sigma=0.005): 
        self.sigma = sigma
        self.transform = transform 
        self.out_res = out_res
        anno_dir = os.path.join(data_dir,"Processing_Results/RGBD/2021_10_Optical_marker_annotation_Hive/Uplifted_with_depth_cam_local") 
        o_img_dir = "/vol/research/animal_motion_data/datasets_internal/SVM_Dogs_2019/Keypoint_Annotation_2021_09/Images" # original joints 
        seg_dir = "/vol/research/animal_motion_data/datasets_internal/SVM_Dogs_2019_Segmentations/" # segmentations
        img_dir = os.path.join(data_dir,"Data/RGBD/Trimmed_Undist_FOV_Opt_2022_01/")
        calib_dir = os.path.join(data_dir, "Metadata_Config/RGBD/Calib/v2022_01_17_undistorted_fov_optimized__per_seq")
    
        self.images = []
        self.segs = []
        self.keypoints = []
        self.visibility = []
        self.cx = []
        self.cy = []
        self.fx = []
        self.fy = []
        self.img_width = []
        self.img_height = []
        
        curr_images = os.listdir(o_img_dir)
        for c in curr_images: 
            sub, trial, pass_id, cam_num, frame = c.split("_")
            n_name = "{}_{}_{}".format(sub,trial,pass_id)
            i_name = "{}_{}{}".format(sub,trial,pass_id)
            text_file = os.path.join(anno_dir,n_name, cam_num+".txt")
            if os.path.exists(text_file):
                # === get the keypoints ===
                with open(text_file, 'r') as f: 
                    lines = f.readlines()[4:]
                    for l in lines:
                        frame_list = l.split()
                        frame_id = frame_list[0]
                        if int(frame_id) ==  int(frame.split('.')[0]):
                            keypoints = np.array(frame_list[1:],dtype=np.float32).reshape(-1,3)
                            if (3==sum(np.isnan(keypoints[2]))):
                                break
                            vis = np.array([int(~np.isnan(v).any()) for v in keypoints])
                            keypoints[np.isnan(keypoints)] = 0
                            vis[np.isnan(vis)] = -1
                            if vis[2] == -1 or vis[3] == -1:
                                break
                            # === get calib data === 
                            with open(os.path.join(calib_dir,sub+"_"+trial,"Calib","cameras","cam-{}.json".format(cam_num))) as jsonf:
                                calib_data = json.load(jsonf) 
                            # === get image data === 
                            self.images.append(os.path.join(img_dir,i_name,"Images",cam_num,c.split("_")[-1])) 
                            self.segs.append(os.path.join(seg_dir,i_name,'Images', cam_num,c.split("_")[-1]))
                            self.keypoints.append(keypoints)
                            self.visibility.append(vis)
                            self.cx.append(float(calib_data['camera']['cx']))
                            self.cy.append(float(calib_data['camera']['cy']))
                            self.fx.append(float(calib_data['camera']['fx']))
                            self.fy.append(float(calib_data['camera']['fy']))
                            self.img_width.append(int(calib_data['camera']['width']))
                            self.img_height.append(int(calib_data['camera']['height']))
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx): 
        
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        seg = np.array(Image.open(self.segs[idx]).convert('L'))
         
        kpts = np.array(self.keypoints[idx])
        vis = np.array(self.visibility[idx])
        x = np.zeros(32)
        y = np.zeros(32)
        z = np.zeros(32)
        v = np.ones(32) * -1
        x_temp = kpts[:,0]
        y_temp = kpts[:,1]
        z_temp = kpts[:,2]

        k_indices = np.array([2,-1,-1,1,-1,-1,6,7,8,9,-1,13,14,15,16,18,20,21,22,24,26,27,28,-1,-1,-1,-1,-1,-1,-1,-1,-1])
        for iidx, ind in enumerate(k_indices): 
            if ind != -1: 
                x[iidx] = x_temp[ind]
                y[iidx] = y_temp[ind] 
                z[iidx] = z_temp[ind]
                v[iidx] = vis[ind]               
        x = torch.from_numpy(x) * 1000 
        y = torch.from_numpy(y) * 1000 
        z = torch.from_numpy(z) * 1000 
        vis = torch.from_numpy(v) 

        xyz = torch.stack((x,y,z), dim=1) 
        cx = self.cx[idx]
        cy = self.cy[idx]
        fx = self.fx[idx]
        fy = self.fy[idx]
        n_w = self.img_width[idx]
        n_h = self.img_height[idx]

        # === normalise the keypoints ===
        x3d, y3d, z3d = normalise_skeleton(xyz, cx, cy, n_w, n_h, fx, fy=fy)
        x3d = torch.as_tensor(x3d)
        y3d = torch.as_tensor(y3d)
        z3d = torch.as_tensor(z3d)

        xy_plane = torch.stack((x3d,y3d), dim=-1)
        zy_plane = torch.stack((z3d,y3d), dim=-1)
        xz_plane = torch.stack((x3d,z3d), dim=-1)

        target = torch.zeros(len(x3d)*3, self.out_res, self.out_res, dtype=torch.float32)
        vis[vis<0] = 0
        for k_idx in range(len(x3d)): 
            if vis[k_idx] > 0: # if keypoint is visible
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
                
        seg_o = transforms.ToPILImage()(seg)
        seg = transforms.Resize((448, 448))(seg_o)
        seg_area = transforms.ToTensor()(seg).reshape(448,448)
        (sy, sx) = np.where(seg_area > 0) 
        smin_x, smin_y = sx.min(), sy.min() 
        smax_x, smax_y = sx.max(), sy.max()  
        s_w = smax_x - smin_x  
        s_h = smax_y - smin_y 
        seg_area = s_w*s_h

        seg = transforms.Resize((32, 32))(seg_o)
        seg = transforms.ToTensor()(seg)
        if self.transform is not None :
            image = self.transform(image)

        final_3dkpts = [item for sublist in zip(x3d, y3d, z3d) for item in sublist] # 96
        target = {'keypoints3d': torch.as_tensor(final_3dkpts).reshape(-1,3), 
                'cx': self.cx[idx],
                'cy': self.cy[idx],
                'hmps': target.to(torch.float32),
                'visiblity': vis,
                'nw': self.img_width[idx],
                'nh': self.img_height[idx],
                'seg_area': seg_area,
                'seg': seg}
        return image, target


if __name__ == "__main__":
    dataset=Svm(transform=None, sigma=0.005)
    img, target = dataset[0] 
