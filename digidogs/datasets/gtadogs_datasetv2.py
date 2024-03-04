import sys
import os 
import json
import torch
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset 
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import random
import kornia as K 
import pickle
from digidogs.configs.defaults import DEFAULT_SKEL
from digidogs.utils.normaliser import normalise_skeleton
from digidogs.rgbddog import utils

def get_3x4RT_matrix_from_blender(location, rotation, isCam=False):
    R_BlenderView_to_OpenCVView = np.diag([1 if isCam else -1,-1,-1])
    R_BlenderView = rotation.T
    T_BlenderView = -1.0 * R_BlenderView @ location
    R_OpenCV = R_BlenderView_to_OpenCVView @ R_BlenderView
    T_OpenCV = R_BlenderView_to_OpenCVView @ T_BlenderView
    RT_OpenCV = np.column_stack((R_OpenCV, T_OpenCV))
    return RT_OpenCV, R_OpenCV, T_OpenCV

class GTADogs(Dataset): 
    
    def __init__(self, data_folder=None, anno_dir=None, data_type="train", inp_res=256, out_res=64, sigma=1, transform=None, patch_size=32):

        self.data_folder = data_folder 
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.transform = transform
        self.patch_size = patch_size
        if anno_dir is not None: 
            self.anno_dir = anno_dir
        else:
            self.anno_dir = data_folder

        if data_type == "train":
            with open(os.path.join(self.anno_dir,"split/trainpose.pickle"), 'rb') as json_file:
                self.dataset = pickle.load(json_file)
        elif data_type == "val":
            with open(os.path.join(self.anno_dir,"split/valpose.pickle"),'rb') as json_file:
                self.dataset = pickle.load(json_file) 
        else: 
            with open(os.path.join(self.anno_dir,"split/testpose.pickle"),'rb') as json_file:
                self.dataset = pickle.load(json_file) 
    
        dataset_len = len(self.dataset['images'])
        self.images = [self.dataset['images'][i]['file_path'] for i in range(dataset_len)]
        self.annotations = [self.dataset['annotations'][i] for i in range(dataset_len)]
        # '/media/ms02373/TOSHIBA EXT/SteamLibrary/steamapps/common/Grand Theft Auto V/PoseData/08_30_2022/00001/MotionSequence.bvh'

    def __getitem__(self, idx): 
        # dict_keys(['info', 'images', 'annotations', 'categories', 'licenses'])
        # === EXTRACT IMAGE INFORMATION ===
        img_pth = os.path.join(self.data_folder, self.images[idx]) 
        img_pil = Image.open(img_pth)
        img_pilOG = img_pil.convert('RGB')
        #img = np.array(img_pil.convert("RGB")).astype(np.float32) # convert to numpy array  
        #img = img / img.max()
        #if img.max() > 1 and img.max() <=255: 
        #    img /= 255
        img_pil.seek(2)
        # Convert the image to floating-point representation
        image_float = img_pil.point(lambda p: p / 255.0, 'F')
        img_pil.close()

        # Normalize the pixel values to the range [0, 1]
        min_value = float(image_float.getextrema()[0])
        max_value = float(image_float.getextrema()[1])
        s_range = max_value - min_value
        if s_range == 0: 
            s_range = 1
        seg_map = np.array(image_float.point(lambda p: (p - min_value) / (s_range)))
        seg_map[seg_map>0.5] = 1
        seg_map = seg_map.astype(np.uint8)

        # === BBOX INFORMATION === 
        (y,x) = np.where(seg_map==seg_map.max())

        minx, miny, maxx, maxy = x.min(), y.min(), x.max(), y.max()  
        bbox_center = [(minx+maxx)*0.5,(miny+maxy)*0.5] 
        
        # === GET THE SIZE OF THE DOG BIG/SMALL === 
        big_dog = 1 # 0 = small, 1 = big
        curr_dog_type = self.annotations[idx]["dog_type"]
        if curr_dog_type  == "a_c_westy" or curr_dog_type == "a_c_pug" or curr_dog_type == "a_c_poodle": 
            big_dog = 0
    
        # === EXTRACT THE KEYPOINT INFORMATION === 
        kpts = self.annotations[idx]["keypoints"][6:] # we are extracting the root and something else 
        x = np.array(kpts[::3])
        y = np.array(kpts[1::3])
        v = np.array(kpts[2::3]) 

        # === EXTRACT THE 3D KEYPOINT INFORMATION ===
        kpts_3d = np.array(self.annotations[idx]['keypoints3d'][6:]) 
        x3d = kpts_3d[::3]
        y3d = kpts_3d[1::3]
        z3d = kpts_3d[2::3]
        campos = np.array(self.annotations[idx]['campos'])
        camrot = np.array(self.annotations[idx]['camrot'])
        r = R.from_euler('xyz', camrot, degrees=True).as_matrix()
        L = np.eye(4)
        L[:3,:3] = r
        L[:3,3] = campos
        radians = 50 * 3.14 / 180.0;
        cotangent = 1 / np.tan(radians/2)
        focal_length = (720/2) * cotangent # mm) 
        
        # visibility 0 = not visible, 1=in image but not visible, 2=visible 
        if curr_dog_type == "a_c_rottweiler": # taking the long tail off..
            v[36:39] = [0] * 3
        #if curr_dog_type == "a_c_pug": 
        #    v[33] = 0

        nparts = len(x)
        for k in range(nparts):
            # if both -1 then false
            if x[k] < 0 and y[k] < 0: 
                v[k] = 0
            # if invisible (self-occlusion) -> visible
            if v[k] == 1: 
                v[k]=2

        # change the 2 to 1 
        for k in range(nparts):
            if v[k] == 2: 
                v[k] = 1

        # original keypoints is 39 
        if big_dog == 0: 
            x =(x[[0,2,5,6,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]]).tolist()
            y =(y[[0,2,5,6,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]]).tolist()
            v =(v[[0,2,5,6,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]]).tolist()
            x3d =(x3d[[0,2,5,6,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]]).tolist()
            y3d =(y3d[[0,2,5,6,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]]).tolist()
            z3d =(z3d[[0,2,5,6,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]]).tolist()
        else:
            x =(x[[0,1,4,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,35,37,38]]).tolist()
            y =(y[[0,1,4,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,35,37,38]]).tolist()
            v =(v[[0,1,4,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,35,37,38]]).tolist()
            x3d =(x3d[[0,1,4,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,35,37,38]]).tolist()
            y3d =(y3d[[0,1,4,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,35,37,38]]).tolist()
            z3d =(z3d[[0,1,4,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,25,26,27,28,30,35,37,38]]).tolist()

        # padd to 
        while(len(x) != 32):
            x.append(0)
            y.append(0)
            v.append(0)
            x3d.append(0)
            y3d.append(0)
            z3d.append(0)

        # 2d keypoints.. but do we really need those?  
        x = torch.as_tensor(x,dtype=torch.float32)
        y = torch.as_tensor(y,dtype=torch.float32)
        v = torch.as_tensor(v,dtype=torch.float32)

        # 3d camera coordinates 
        x3d = torch.as_tensor(x3d,dtype=torch.float32) * 1000
        y3d = torch.as_tensor(y3d,dtype=torch.float32) * 1000
        z3d = torch.as_tensor(z3d,dtype=torch.float32) * 1000

        # normalize 
        #limb_lenghts = [] 
        #for limb in DEFAULT_SKEL: 
        #    ind1 = limb[0]
        #    ind2 = limb[1]
        #    pnt1x = x3d[ind1] 
        #    pnt1y = y3d[ind1] 
        #    pnt1z = z3d[ind1] 
        #    pnt2x = x3d[ind2] 
        #    pnt2y = y3d[ind2] 
        #    pnt2z = z3d[ind2] 
        #
        #    length = np.linalg.norm(np.array([pnt2x,pnt2y,pnt2z])-np.array([pnt1x, pnt1y, pnt1z]))
        #    limb_lenghts.append(length)
    
        #sum_square_limb_length =np.sum(np.square(limb_lenghts)) 
        #x3d = x3d / sum_square_limb_length
        #y3d = y3d / sum_square_limb_length
        #z3d = z3d / sum_square_limb_length

        # === AUGMENTATION ===
        #img = img.astype('float32')

        # === CROP IMAGE AND SET TO 256 X 256 === 
        scale_crop = random.randint(1,720//4) 
        #scale_crop = 1
        xmin = minx -scale_crop
        ymin = miny -scale_crop
        xmax = maxx + scale_crop
        ymax = maxy + scale_crop
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

        # what about padding ? 
        cx = 1280 / 2 - xcropmin
        cy = 720 / 2 - ycropmin
        crop_image = img_pilOG.crop((xcropmin, ycropmin, xcropmax, ycropmax))
        crop_seg = Image.fromarray(seg_map) 
        crop_seg = crop_seg.crop((xcropmin, ycropmin, xcropmax, ycropmax))
        #cropped_padded_img = Image.new('RGB', (crop_width, crop_height), (0,0,0))
        #cropped_padded_img.paste(crop_image, (0,0))
        #cropped_image = np.array(cropped_padded_img) 
        cropped_image = np.array(crop_image)
        cropped_seg = np.array(crop_seg,dtype=np.uint8)
        #cropped_image = img[ycropmin:ycropmax,xcropmin:xcropmax]

        img_pilOG.close()
        n_h, n_w, _ = cropped_image.shape
        #cropped_seg = seg_map[ycropmin:ycropmax,xcropmin:xcropmax] # we should include the background

        # --- relative ---
        root = [x3d[0], y3d[0], z3d[0]]
        xrel = x3d - x3d[0]
        yrel = y3d - y3d[0]
        zrel = z3d - z3d[0]

        # === normalise ===
        # https://en.wikipedia.org/wiki/Z-buffering
        pts3d = np.stack((x3d.numpy(), y3d.numpy(), z3d.numpy()),1).reshape(32,3)
        hom_pos3d = np.hstack((pts3d, np.ones((pts3d.shape[0],1))))
        world_rotations = np.eye(3) # blender word rotations of keypoints
        world_positions = np.matmul(L,hom_pos3d.T).T#[:,:3] # blender world positions of keypoints..  
        RT_camMatrix = get_3x4RT_matrix_from_blender(campos, L[:3,:3], isCam=True)
        w_pos = []
        for l in world_positions[:,:3]:
            w_pos.append(get_3x4RT_matrix_from_blender(l, world_rotations, isCam=False)[2])
        w_pos = np.array(w_pos)
        w_pos = np.hstack((w_pos, np.ones((w_pos.shape[0],1))))
        R_, T_ = RT_camMatrix[1:]
        world2cam = np.eye(4)
        world2cam[:3,:3] = R_
        world2cam[:3,3] = T_
        cam_positions = (world2cam @ w_pos.T ).T 
        
        norm_length = np.linalg.norm(cam_positions[2][:3] - cam_positions[0][:3])
        x3d,y3d,z3d = normalise_skeleton(cam_positions, cx, cy, n_w, n_h, focal_length)
        x3d = torch.from_numpy(x3d) 
        y3d = torch.from_numpy(y3d) 
        z3d = torch.from_numpy(z3d) 
        xy_plane = torch.stack((x3d,y3d), dim=-1)
        zy_plane = torch.stack((z3d,y3d), dim=-1)
        xz_plane = torch.stack((x3d,z3d), dim=-1)
        ## === GENERATE GROUND TRUTH ===
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
                
        seg_ta = transforms.Compose([transforms.ToPILImage(), transforms.Resize((448,448)), transforms.ToTensor()])
        seg_area = seg_ta(cropped_seg*255).numpy().reshape((448,448))
        (sy, sx) = np.where(seg_area > 0) 
        smin_x, smin_y = sx.min(), sy.min() 
        smax_x, smax_y = sx.max(), sy.max()  
        s_w = smax_x - smin_x  
        s_h = smax_y - smin_y 
        seg_area = np.sqrt(s_w*s_h)

        seg_t = transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.out_res,self.out_res)), transforms.ToTensor()])
        cropped_seg = seg_t(cropped_seg*255)
        if type(self.transform) == transforms.transforms.Compose:
            #cropped_image *= 255
            cropped_image = cropped_image.astype('uint8')
            cropped_image = self.transform(cropped_image)

        final_3dkpts = [item for sublist in zip(x3d, y3d, z3d) for item in sublist] # 96
        final_3drel = [item for sublist in zip(xrel, yrel, zrel) for item in sublist]
        root = torch.FloatTensor(root) #* (1280/focal_length)

        target = {
                'keypoints3d':torch.FloatTensor(final_3dkpts), 
                'cx': cx, 
                'cy': cy, 
                'hmps': target.to(torch.float32),
                'visiblity' : v.to(torch.float32), 
                'nw': n_w,
                'nh': n_h, 
                'seg': torch.Tensor(cropped_seg),
                'seg_area': seg_area,
                'norm_3d': norm_length
                }

        # === ENABLE TEST ===
        if torch.sum(v) != 26:
            replacemend_index = max(0,idx-1)
            cropped_img ,target = self.__getitem__(replacemend_index)
        return cropped_image, target

    def __len__(self): 
        return len(self.dataset["images"])

