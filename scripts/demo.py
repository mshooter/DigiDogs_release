import sys
import torch 
import random
from PIL import Image 
import numpy as np
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from digidogs.datasets.datamoduleposev2 import GtaModule
from digidogs.datasets.gtadogs_datasetv2 import GTADogs
from digidogs.datasets.stanext import StanExt
from digidogs.datasets.rgbddog import RgbdDog
from digidogs.datasets.demo_dataset import DemoDataset
from digidogs.datasets.svm import Svm
from digidogs.models.litposev2 import LitPoser
from digidogs.configs.defaults import DEFAULT_SKEL, SKEL_COLORS
from digidogs.utils.normaliser import denormalise_skeleton

if __name__ == "__main__":

    type_data = sys.argv[1] # type of dataset
    if type_data == 'custom':
        demo_dir = sys.argv[2] 
    print(demo_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    COLOR_G = 'limegreen'
    COLOR_P = 'deeppink'
    transform_dino = transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.Resize((448,448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=MEAN, std=STD)
                                         ])

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [1/STD[0], 1/STD[1], 1/STD[2]]),
                                    transforms.Normalize(mean = [-MEAN[0], -MEAN[1], -MEAN[2]],
                                                     std = [ 1., 1., 1. ]),
                                    transforms.ToPILImage()])

    if type_data == 'stanford':
        dataset = StanExt(data_dir="/vol/research/animal_motion_data/datasets_3rd_party/Stanford_Dogs", data_type='test', transform=transform_dino, sigma=0.005)
    elif type_data == "gta":
        dataset = GTADogs(data_folder="/vol/research/datasets/mixedmode/gtadogs", anno_dir='/scratch/DigiDogs/data', data_type="test", out_res=32, sigma=0.005, transform=transform_dino)
    elif type_data == 'svm': 
        dataset=Svm(data_dir="/vol/research/animo/datasets_internal/dogs_svm_mars_2019/", transform=transform_dino, sigma=0.005)
    elif type_data == 'rgbd':
        dataset = RgbdDog(data_dir="/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ms02373/", train_type="test", test_dog='dog2', transform=transform_dino)
    elif type_data == 'custom': 
        dataset = DemoDataset(demo_dir, transform=transform_dino)

    LEN_DATASET = len(dataset)
    BATCH_SIZE = 1
    IMG_SIZE = 448
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) 
    # === checkpoint change === 
    # ./checkpoints/rgbddog_poseF16/epoch=11_val_loss=-0.0299.ckpt
    # ./checkpoints/svm_poseF16/epoch=11_val_loss=-1.1653.ckpt
    # ./checkpoints/gta_poseF16/epoch=15_val_loss=0.4664.ckpt
    model = LitPoser.load_from_checkpoint("./checkpoints/gta_poseF16/epoch=15_val_loss=0.4664.ckpt") 
    model.to(device) 
    model.eval()

    for index, (images, target) in enumerate(loader): 

        if type_data != 'custom':
            vis = target['visiblity'] 
            hmps = target['hmps']
            gt_xyz = model._compute_coords(hmps[:,::3], hmps[:,1::3], hmps[:,2::3])

        cx = target['cx']
        cy = target['cy']
        nw = target['nw']
        nh = target['nh']

        # == predict and compute coords ==
        images = images.to(device)
        hxy, hzy, hxz,_ = model(images) 
        xyz = model._compute_coords(hxy, hzy, hxz) 

        for b in range(BATCH_SIZE):
            img = invTrans(images[b])
            if type_data != 'custom':
                gx, gy, gz = denormalise_skeleton(gt_xyz[b] , cx[b], cy[b], nw[b], nh[b])
                gt_kpts = torch.stack((gx, gy, gz), dim=1)[:26] 
                gt_proj_pts = torch.stack((gx*IMG_SIZE/nw[b], gy*IMG_SIZE/nh[b]), dim=1).detach().cpu().numpy()
                gt_kpts = (gt_kpts - gt_kpts[0]).detach().cpu()

            # === denormalise ===
            px, py, pz = denormalise_skeleton(xyz[b] , cx[b], cy[b], nw[b], nh[b])
            kpts = torch.stack((px, py, pz), dim=1)[:26]  
            proj_pts = torch.stack((px*IMG_SIZE/nw[b], py*IMG_SIZE/nh[b]), dim=1).detach().cpu().numpy()
            kpts = (kpts - kpts[0]).detach().cpu()

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.axis('off')
            ax2 = fig.add_subplot(122,projection='3d')
            ax2.invert_yaxis()
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.view_init(elev=1, azim=0, vertical_axis='y')
            ax2.set_box_aspect([1, 1, 1])
   
            # display image
            ax1.imshow(img)
            if type_data != 'custom':
                visibility = vis[b][:26]

            for idx in range(len(kpts)): 
                if type_data == 'svm':
                    ax1.scatter(proj_pts[idx,0],proj_pts[idx,1],c=COLOR_P, s=5)
                    ax2.scatter(kpts[idx,0],kpts[idx,1], kpts[idx,2],c=COLOR_P, s=5)
                    if visibility[idx] > 0:
                        ax1.scatter(gt_proj_pts[idx,0],gt_proj_pts[idx,1],c=COLOR_G, s=5)
                        ax2.scatter(gt_kpts[idx,0],gt_kpts[idx,1],gt_kpts[idx,2],c=COLOR_G, s=5)
                elif type_data == 'custom':
                    ax1.scatter(proj_pts[idx,0],proj_pts[idx,1],c=COLOR_P, s=5)
                    ax2.scatter(kpts[idx,0],kpts[idx,1], kpts[idx,2],c=COLOR_P, s=5)
                else: 
                    if visibility[idx] > 0:
                        ax1.scatter(gt_proj_pts[idx,0],gt_proj_pts[idx,1],c=COLOR_G, s=5)
                        ax1.scatter(proj_pts[idx,0],proj_pts[idx,1],c=COLOR_P, s=5)
                        ax2.scatter(kpts[idx,0],kpts[idx,1], kpts[idx,2],c=COLOR_P, s=5)
                        ax2.scatter(gt_kpts[idx,0],gt_kpts[idx,1],gt_kpts[idx,2],c=COLOR_G, s=5)

            for idx, s in enumerate(DEFAULT_SKEL): 
                joint1, joint2 = s 
                if joint1 < 26 and joint2 < 26:
                    if type_data == 'svm':
                        ax1.plot([proj_pts[joint1,0], proj_pts[joint2,0]], [proj_pts[joint1,1],proj_pts[joint2,1]], c=COLOR_P)
                        ax2.plot([kpts[joint1,0],kpts[joint2,0]], [kpts[joint1,1],kpts[joint2,1]], [kpts[joint1,2],kpts[joint2,2]],c=COLOR_P)
                        if(visibility[joint1] > 0 and visibility[joint2] > 0):  
                            ax1.plot([gt_proj_pts[joint1,0], gt_proj_pts[joint2,0]], [gt_proj_pts[joint1,1],gt_proj_pts[joint2,1]], c=COLOR_G)
                            ax2.plot([gt_kpts[joint1,0],gt_kpts[joint2,0]], [gt_kpts[joint1,1],gt_kpts[joint2,1]], [gt_kpts[joint1,2],gt_kpts[joint2,2]],c=COLOR_G)
                    elif type_data == 'custom':
                        ax1.plot([proj_pts[joint1,0], proj_pts[joint2,0]], [proj_pts[joint1,1],proj_pts[joint2,1]], c=COLOR_P)
                        ax2.plot([kpts[joint1,0],kpts[joint2,0]], [kpts[joint1,1],kpts[joint2,1]], [kpts[joint1,2],kpts[joint2,2]],c=COLOR_P)
                    else:
                        if(visibility[joint1] > 0 and visibility[joint2] > 0):  
                            ax1.plot([gt_proj_pts[joint1,0], gt_proj_pts[joint2,0]], [gt_proj_pts[joint1,1],gt_proj_pts[joint2,1]], c=COLOR_G)
                            ax1.plot([proj_pts[joint1,0], proj_pts[joint2,0]], [proj_pts[joint1,1],proj_pts[joint2,1]], c=COLOR_P)
                            ax2.plot([kpts[joint1,0],kpts[joint2,0]], [kpts[joint1,1],kpts[joint2,1]], [kpts[joint1,2],kpts[joint2,2]],c=COLOR_P)
                            ax2.plot([gt_kpts[joint1,0],gt_kpts[joint2,0]], [gt_kpts[joint1,1],gt_kpts[joint2,1]], [gt_kpts[joint1,2],gt_kpts[joint2,2]],c=COLOR_G)
            plt.show()
            plt.close()
            print(f"Image: {index+1} out of {LEN_DATASET}")
        #if index == 0:
        #    break
