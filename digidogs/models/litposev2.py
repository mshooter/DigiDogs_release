import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lightning.pytorch as pl 
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import kornia as K 
import torchvision.transforms as transforms
import torchmetrics
from digidogs.models.posev2 import SimplePose
from digidogs.utils.loss import LocationmapLoss,MaskedMSELoss, js_div_loss_2d_masked 
from digidogs.utils.metrics import rigid_align, metrics2d,batch_compute_similarity_transform_torch,apply_alignment 
from digidogs.configs.defaults import DEFAULT_SKEL, SKEL_COLORS
from digidogs.utils.normaliser import denormalise_skeleton

COLOR_G = 'limegreen'
COLOR_P = 'deeppink'
LINEWIDTH = 1

def make_3d(fig, n_figs, i, b):
    ax2 = fig.add_subplot(n_figs,4,b*4+i,projection='3d')
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.zaxis.set_ticklabels([])
    for line in ax2.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax2.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax2.zaxis.get_ticklines():
        line.set_visible(False)
    #ax2.set_xlabel('X')
    #ax2.set_ylabel('Y')
    #ax2.set_zlabel('Z')
    ax2.set_box_aspect([1, 1, 1])
    ax2.invert_yaxis()
    return ax2

class LitPoser(pl.LightningModule): 
    def __init__(self, 
            unfreeze=True, 
            learning_rate=1e-3, 
            batch_size=16,
            n_keypoints=32,
            sched='plateau',
            pct_start=0.1,
            div_factor=25,
            final_div_factor=10000):
        super().__init__() 
        self.pct_start=pct_start
        self.div_factor=div_factor
        self.final_div_factor=final_div_factor
        self.validation_step_outputs = []
        self.sched = sched
        self.n_keypoints = n_keypoints
        self.model = SimplePose(unfreeze=unfreeze, 
                              n_keypoints=n_keypoints) # there are more params
        self.learning_rate = learning_rate 
        self.batch_size = batch_size 
        self.mse_loss = MaskedMSELoss()
        self.bceloss = nn.BCEWithLogitsLoss()
        self.iou = torchmetrics.classification.BinaryJaccardIndex()
        self.img_size = 448
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _compute_coords(self, h_xy, h_zy, h_xz):
        """
        heatmaps (tensor): contains the 2D predicted heatmaps (gaussian)
        """
        h_xy = K.geometry.subpix.spatial_expectation2d(h_xy)  # B, K, 2  
        h_zy = K.geometry.subpix.spatial_expectation2d(h_zy)  # B, K, 2  
        h_xz = K.geometry.subpix.spatial_expectation2d(h_xz)  # B, K, 2  
        z = 0.5 * (h_zy[:,:,0:1] + h_xz[:,:,1:2])
        return torch.cat([h_xy,z], -1)

    def _compute_loss(self, batch, is_3d=True):
        images, target = batch 
        gt_H = target['hmps'] 
        vis = target['visiblity']
        cx = target['cx'].reshape(self.batch_size,1)
        cy = target['cy'].reshape(self.batch_size,1)
        nw = target['nw'].reshape(self.batch_size,1)
        nh = target['nh'].reshape(self.batch_size,1)

        # === predictions from model ===
        pred_Hxy, pred_Hzy, pred_Hxz, pred_seg = self.model(images)

        # === 2d jse loss and euclidean loss ===
        jsd2d = js_div_loss_2d_masked(pred_Hxy, gt_H[:,0::3],vis) # xy
        pred_xyz = self._compute_coords(pred_Hxy, pred_Hzy, pred_Hxz)
        gt_xyz = self._compute_coords(gt_H[:,0::3], gt_H[:,1::3], gt_H[:,2::3])
        loss2d = jsd2d + self.mse_loss(pred_xyz[:,:,:2].clone(), gt_xyz[:,:,:2].clone(), vis) 
        loss = loss2d

        # === compute the 3d loss === 
        loss3d = 0
        if is_3d:
            loss3d += self.mse_loss(pred_xyz, gt_xyz, vis) 
            loss3d += js_div_loss_2d_masked(pred_Hxy, gt_H[:,0::3],vis) 
            loss3d += js_div_loss_2d_masked(pred_Hzy, gt_H[:,1::3],vis) 
            loss3d += js_div_loss_2d_masked(pred_Hxz, gt_H[:,2::3],vis) 
            loss = loss3d

        # === compute the segmentation loss === 
        gt_seg = target['seg']
        seg_loss = self.bceloss(pred_seg.float(), gt_seg.float())
        #self.log_dict({'2dloss':loss2d, '3dloss':loss3d})
        loss += seg_loss
        return loss, images, gt_xyz, pred_xyz, vis, nw, nh,cx,cy, gt_seg, pred_seg

    def _log_images(self, 
            images, 
            gt_xyz,
            pred_xyz,
            vis,
            nw, 
            nh,
            cx,
            cy,
            gt_seg,
            pred_seg,
            pck=None,
            mpjpe=None,
            aligned=None,
            mean=(0.485, 0.456, 0.406), 
            std =(0.229, 0.224, 0.225), 
            type_step = 'train',
            n_figs=4,
            step=None,
            align_root=False): 

        tensorboard = self.logger.experiment

        # === IMAGE CONVERSION === 
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [1/std[0], 1/std[1], 1/std[2]]),
                                        transforms.Normalize(mean = [-mean[0], -mean[1], -mean[2]],
                                                         std = [ 1., 1., 1. ])])
        images = invTrans(images) 
        vis = vis.detach().cpu().numpy()
        nw = nw
        nh = nh
        cx = cx
        cy = cy
        pred_seg = torch.sigmoid(pred_seg) 
        pred_seg = pred_seg > 0.5
        pred_seg = pred_seg.type(torch.uint8)
        gt_seg = transforms.Resize((448,448))(gt_seg)
        pred_seg = transforms.Resize((448,448))(pred_seg)
        # === PLOT FIGURE ===
        fig = plt.figure()

        for b in range(n_figs):
            ax1 = fig.add_subplot(n_figs,4,b*4+1)
            ax3 = fig.add_subplot(n_figs,4,b*4+2)
            ax1.axis('off')
            ax3.axis('off')

            p_seg = transforms.ToPILImage()(pred_seg[b])
            g_seg = transforms.ToPILImage()(gt_seg[b])
            img = images[b] 
            img = ((img / img.max()) * 255).to(torch.uint8).to('cpu')
            img = transforms.ToPILImage()(img)

            ax1.imshow(img)
            if pck is not None:
                ax1.set_title(f'PCK: {pck[b]:.2f} \nMPJPE: {mpjpe[b]:.2f}')
            #ax1.imshow(g_seg, cmap='jet',alpha=0.5)
            #ax3.imshow(p_seg, cmap='jet')

            gx, gy, gz = denormalise_skeleton(gt_xyz[b] , cx[b], cy[b], nw[b], nh[b])
            px, py, pz = denormalise_skeleton(pred_xyz[b], cx[b], cy[b], nw[b], nh[b])
            pts3d_gt = torch.stack((gx, gy, gz), dim=1) # 3d coords 
            pts3d_pred = torch.stack((px, py, pz), dim=1)  
            gt_proj_pts = torch.stack((gx*self.img_size/nw[b], gy*self.img_size/nh[b]), dim=1).detach().cpu().numpy() # projections 
            pred_proj_pts = torch.stack((px*self.img_size/nw[b], py*self.img_size/nh[b]), dim=1).detach().cpu().numpy() 

            # === 2D ground truth and predictions === 
            for idx in range(len(gt_proj_pts)):
                #if pck is not None:
                #    ax1.text(100, 100, f'PCK{pck[b]:.2f} \n MPJPE{mpjpe[b]:.2f}', fontsize=8, fontweight='bold', color='white')

                ax1.scatter(pred_proj_pts[idx,0], pred_proj_pts[idx,1], c=COLOR_P, s=2)  
                if vis[b][idx]>0: 
                    ax1.scatter(gt_proj_pts[idx,0], gt_proj_pts[idx,1], c=COLOR_G, s=2, marker='.')  


                for c in DEFAULT_SKEL: 
                    joint1, joint2 = c 
                    if joint1 < 26 and joint2 < 26:
                        ax1.plot([pred_proj_pts[joint1,0], pred_proj_pts[joint2,0]],[pred_proj_pts[joint1,1], pred_proj_pts[joint2,1]], c=COLOR_P) 
                        if vis[b][joint1] > 0 and vis[b][joint2] > 0:
                            ax1.plot([gt_proj_pts[joint1,0], gt_proj_pts[joint2,0]],[gt_proj_pts[joint1,1], gt_proj_pts[joint2,1]], c=COLOR_G) 


            # === 3d ground truth and predictions === 
            pts3d_gt = pts3d_gt.detach().cpu().numpy()
            pts3d_pred = pts3d_pred.detach().cpu().numpy()
            if align_root:
                pts3d_gt = pts3d_gt - pts3d_gt[0]
                pts3d_pred = pts3d_pred - pts3d_pred[0]
                #aligned = aligned - aligned[0]
            ax2 = make_3d(fig, n_figs, 2, b)
            ax2.view_init(elev=30, azim=0, vertical_axis='y') 
            ax4 = make_3d(fig,n_figs,3,b)
            ax4.view_init(elev=30, azim=45, vertical_axis='y') 
            ax3 = make_3d(fig,n_figs,4,b)
            ax3.view_init(elev=30, azim=90, vertical_axis='y') 
            SIZE = 3
            for idx in range(len(pts3d_gt)):
                ax2.scatter(pts3d_pred[idx,0], pts3d_pred[idx,1], pts3d_pred[idx,2], c=COLOR_P,s=SIZE)
                ax4.scatter(pts3d_pred[idx,0], pts3d_pred[idx,1], pts3d_pred[idx,2], c=COLOR_P,s=SIZE)
                ax3.scatter(pts3d_pred[idx,0], pts3d_pred[idx,1], pts3d_pred[idx,2], c=COLOR_P,s=SIZE)

                if vis[b][idx] > 0:
                    ax2.scatter(pts3d_gt[idx,0], pts3d_gt[idx,1], pts3d_gt[idx,2], c=COLOR_G,s=SIZE)
                    ax4.scatter(pts3d_gt[idx,0], pts3d_gt[idx,1], pts3d_gt[idx,2], c=COLOR_G,s=SIZE)
                    ax3.scatter(pts3d_gt[idx,0], pts3d_gt[idx,1], pts3d_gt[idx,2], c=COLOR_G,s=SIZE)

                    if aligned is not None:
                        ax2.scatter(aligned[b,idx,0], aligned[b,idx,1], aligned[b,idx,2], c=COLOR_P,s=SIZE)
                        ax4.scatter(aligned[b,idx,0], aligned[b,idx,1], aligned[b,idx,2], c=COLOR_P,s=SIZE)
                        ax3.scatter(aligned[b,idx,0], aligned[b,idx,1], aligned[b,idx,2], c=COLOR_P,s=SIZE)
        
            for c in DEFAULT_SKEL: 
                joint1, joint2 = c 
                if joint1 < 26 and joint2 < 26:
                    ax2.plot([pts3d_pred[joint1,0], pts3d_pred[joint2,0]],[pts3d_pred[joint1,1], pts3d_pred[joint2,1]],[pts3d_pred[joint1,2], pts3d_pred[joint2,2]], c=COLOR_P, linewidth=LINEWIDTH) 
                    ax4.plot([pts3d_pred[joint1,0], pts3d_pred[joint2,0]],[pts3d_pred[joint1,1], pts3d_pred[joint2,1]],[pts3d_pred[joint1,2], pts3d_pred[joint2,2]], c=COLOR_P, linewidth=LINEWIDTH) 
                    ax3.plot([pts3d_pred[joint1,0], pts3d_pred[joint2,0]],[pts3d_pred[joint1,1], pts3d_pred[joint2,1]],[pts3d_pred[joint1,2], pts3d_pred[joint2,2]], c=COLOR_P, linewidth=LINEWIDTH) 
                    if vis[b][joint1] > 0 and vis[b][joint2] > 0:

                        ax2.plot([pts3d_gt[joint1,0], pts3d_gt[joint2,0]],[pts3d_gt[joint1,1], pts3d_gt[joint2,1]],[pts3d_gt[joint1,2], pts3d_gt[joint2,2]], c=COLOR_G,linewidth=LINEWIDTH) 
                        ax4.plot([pts3d_gt[joint1,0], pts3d_gt[joint2,0]],[pts3d_gt[joint1,1], pts3d_gt[joint2,1]],[pts3d_gt[joint1,2], pts3d_gt[joint2,2]], c=COLOR_G,linewidth=LINEWIDTH) 
                        ax3.plot([pts3d_gt[joint1,0], pts3d_gt[joint2,0]],[pts3d_gt[joint1,1], pts3d_gt[joint2,1]],[pts3d_gt[joint1,2], pts3d_gt[joint2,2]], c=COLOR_G,linewidth=LINEWIDTH) 

                        if aligned is not None:
                            ax2.plot([aligned[b,joint1,0], aligned[b,joint2,0]],[aligned[b,joint1,1], aligned[b,joint2,1]],[aligned[b,joint1,2], aligned[b,joint2,2]], c=COLOR_P, linewidth=LINEWIDTH) 
                            ax4.plot([aligned[b,joint1,0], aligned[b,joint2,0]],[aligned[b,joint1,1], aligned[b,joint2,1]],[aligned[b,joint1,2], aligned[b,joint2,2]], c=COLOR_P, linewidth=LINEWIDTH) 
                            ax3.plot([aligned[b,joint1,0], aligned[b,joint2,0]],[aligned[b,joint1,1], aligned[b,joint2,1]],[aligned[b,joint1,2], aligned[b,joint2,2]], c=COLOR_P, linewidth=LINEWIDTH) 

        if step is None:
            tensorboard.add_figure('figure-{}'.format(type_step), fig, self.current_epoch)
        else:
            tensorboard.add_figure('figure-{}'.format(type_step), fig, step)
        plt.close() 

    def training_step(self, batch, batch_idx): 
        if isinstance(batch, dict):
            b1 = batch['gta']
            b2 = batch['stanford']
            loss1, images1, gt_xyz1, pred_xyz1,vis1,nw1,nh1,cx1,cy1,gt_seg1,pred_seg1 = self._compute_loss(b1, is_3d=True)
            loss2, images2, gt_xyz2, pred_xyz2,vis2,nw2,nh2,cx2,cy2,gt_seg2,pred_seg2 = self._compute_loss(b2, is_3d=False)
            loss = loss1+loss2
            images = torch.cat([images1, images2],0)
            gt_xyz = torch.cat([gt_xyz1, gt_xyz2],0)
            pred_xyz = torch.cat([pred_xyz1, pred_xyz2],0)
            vis = torch.cat([vis1, vis2],0)
            nw = torch.cat([nw1, nw2],0)
            nh = torch.cat([nh1, nh2],0)
            cx = torch.cat([cx1, cx2],0)
            cy = torch.cat([cy1, cy2],0)
            f = torch.cat([f1, f2],0)
            gt_seg = torch.cat([gt_seg1, gt_seg2],0)
            pred_seg = torch.cat([pred_seg1, pred_seg2],0)
        else:
            loss, images, gt_xyz, pred_xyz,vis,nw,nh,cx,cy,gt_seg,pred_seg = self._compute_loss(batch, is_3d=True)
        self.log('loss/train',loss, batch_size=self.batch_size) 
        if batch_idx == 0:
            self._log_images( 
                             images, 
                             gt_xyz, 
                             pred_xyz, 
                             vis,
                             nw,
                             nh,
                             cx,
                             cy,
                             gt_seg,
                             pred_seg
                             )
        return loss

    def validation_step(self, batch, batch_idx,dataloader_idx=0): 
        if dataloader_idx == 0:
            loss, images, gt_xyz, pred_xyz,vis,nw,nh,cx,cy,gt_seg,pred_seg = self._compute_loss(batch, is_3d=True)
            if batch_idx ==0:
                self._log_images(
                                 images, 
                                 gt_xyz, 
                                 pred_xyz, 
                                 vis,
                                 nw,
                                 nh,
                                 cx,
                                 cy,
                                 gt_seg,
                                 pred_seg,
                                 type_step='val0') 
        elif dataloader_idx == 1: 
            loss, images, gt_xyz, pred_xyz,vis,nw,nh,cx,cy,gt_seg,pred_seg = self._compute_loss(batch, is_3d=True)
            if batch_idx==0:
                self._log_images(
                                 images, 
                                 gt_xyz, 
                                 pred_xyz, 
                                 vis,
                                 nw,
                                 nh,
                                 cx,
                                 cy,
                                 gt_seg,
                                 pred_seg,
                                 type_step='val1') 
        else:
            loss, images, gt_xyz, pred_xyz,vis,nw,nh,cx,cy,gt_seg,pred_seg = self._compute_loss(batch, is_3d=True)
            if batch_idx==0:
                self._log_images(
                                 images, 
                                 gt_xyz, 
                                 pred_xyz, 
                                 vis,
                                 nw,
                                 nh,
                                 cx,
                                 cy,
                                 gt_seg,
                                 pred_seg,
                                 type_step='val') 

        self.validation_step_outputs.append(loss), 
        
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('loss/val',avg_loss, batch_size=self.batch_size, on_step=False, on_epoch=True) 
        self.log('val_loss',avg_loss, batch_size=self.batch_size,logger=False,on_step=False, on_epoch=True) 
        self.validation_step_outputs = []

    def configure_optimizers(self): 
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.sched == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-10, patience=5, factor=0.01)
        elif self.sched == 'multi':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.001)
        else: 
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches, pct_start=self.pct_start, div_factor=self.div_factor, final_div_factor=self.final_div_factor)
        return {'optimizer' : optimizer, 'lr_scheduler':scheduler, 'monitor': 'val_loss'}

    def test_step(self, batch, batch_idx): 
        # compute test projection pck (2D) and pck (3D) and mpjpe in mm 
        images, target = batch 
        seg_area = target['seg_area'].cpu()
        #norm_3d = target['norm_3d'].cpu()
        gt_H = target['hmps'] 
        vis = target['visiblity'].cpu()
        cx = target['cx'].reshape(self.batch_size,1).cpu()
        cy = target['cy'].reshape(self.batch_size,1).cpu()
        nw = target['nw'].reshape(self.batch_size,1).cpu()
        nh = target['nh'].reshape(self.batch_size,1).cpu()
        gt_seg = target['seg']

        # === predictions from model ===
        pred_Hxy, pred_Hzy, pred_Hxz, pred_seg = self.model(images)

        # === access the ground truth & prediction coords ===
        gt_xyz = self._compute_coords(gt_H[:,0::3], gt_H[:,1::3], gt_H[:,2::3]).cpu()
        pred_xyz = self._compute_coords(pred_Hxy, pred_Hzy, pred_Hxz).cpu()

        # === denormalise === 
        d_gts = torch.zeros((self.batch_size, self.n_keypoints, 3)) 
        d_pre = torch.zeros((self.batch_size, self.n_keypoints, 3))
        d_proj_gts = torch.zeros((self.batch_size, self.n_keypoints,2))
        d_proj_pre = torch.zeros((self.batch_size, self.n_keypoints,2))

        for b in range(self.batch_size):
            px, py, pz = denormalise_skeleton(pred_xyz[b], cx[b], cy[b], nw[b], nh[b])
            gx, gy, gz = denormalise_skeleton(gt_xyz[b], cx[b], cy[b], nw[b], nh[b])
            d_pre[b] = torch.stack((px, py, pz), dim=1)  
            d_gts[b] = torch.stack((gx, gy, gz), dim=1) 
            d_proj_gts[b] = torch.stack((gx*self.img_size/nw[b], gy*self.img_size/nh[b]), dim=1) # projections 
            d_proj_pre[b] = torch.stack((px*self.img_size/nw[b], py*self.img_size/nh[b]), dim=1)  
        # === compute the 2d metrpics (mpjpe, pck) ===
        #print(seg_area, 'seg')
        pck2d, mpjpe2d = metrics2d(d_proj_pre[:,:26], d_proj_gts[:,:26], vis[:,:26], norm=seg_area, threshold=0.15)

        #LEGS_INDICES = [5,6,7,8,9,19,11,12,13,14,15,16,17,18,19,20,21,22]
        #TAIL_INDICES = [23,24,25]
        #pck2d_legs, mpjpe2d_legs = metrics2d(d_proj_pre[:,LEGS_INDICES], d_proj_gts[:,LEGS_INDICES], vis[:,LEGS_INDICES], norm=seg_area, threshold=0.15)
        #pck2d_tail, mpjpe2d_tail = metrics2d(d_proj_pre[:,TAIL_INDICES], d_proj_gts[:,TAIL_INDICES], vis[:,TAIL_INDICES], norm=seg_area, threshold=0.15, is_3d=True)
        #
        # === compute the 3d metrics (mpjpe, pck) not aligned ===   
        d_pre_root = d_pre-d_pre[:,0].unsqueeze(1) 
        d_gts_root = d_gts-d_gts[:,0].unsqueeze(1)
        norm = torch.linalg.norm(d_gts_root[:,3] - d_gts_root[:,0], dim=1)
        pck3d, mpjpe3d = metrics2d(d_pre_root[:,:26], d_gts_root[:,:26] ,vis[:,:26], norm=norm, threshold=0.2)

        # === compute the 3d metrics (mpjpe, pck) aligned === #
        d_pre_align = batch_compute_similarity_transform_torch(d_pre_root[:,:26], d_gts_root[:,:26])
        #d_pre_align = torch.from_numpy(d_pre_align).unsqueeze(0)
        pa_pck3d, pa_mpjpe3d = metrics2d(d_pre_align[:,:26], d_gts_root[:,:26], vis[:,:26], norm=norm, threshold=0.2)

        self.log_dict({
                       'pck2d': pck2d.mean(), 
                       'mpjpe2d':mpjpe2d.mean(), 
                       #'pck2d_legs': pck2d_legs, 
                       #'mpjpe2d_legs':mpjpe2d_legs, 
                       #'pck2d_tail': pck2d_tail, 
                       #'mpjpe2d_tails':mpjpe2d_tail 
                        'pck3d':pck3d.mean(), 
                        'mpjpe3d': mpjpe3d.mean(),
                        'pa pck3d': pa_pck3d.mean(),
                        'pa mpjpe3d': pa_mpjpe3d.mean()
                       })

        # visualise 250
        if batch_idx % 5 == 0 : 
            print(
                       'pck2d', pck2d, 
                       'mpjpe2d',mpjpe2d, 
                       #'pck2d_legs', pck2d_legs, 
                       #'mpjpe2d_legs',mpjpe2d_legs, 
                       #'pck2d_tail', pck2d_tail, 
                       #'mpjpe2d_tails',mpjpe2d_tail 
                       'pck3d',pck3d, 
                       'mpjpe3d',mpjpe3d,
                       'pa pck3d', pa_pck3d,
                       'pa mpjpe3d', pa_mpjpe3d
                       )
            self._log_images( 
                            images, 
                            gt_xyz[:,:26], 
                            pred_xyz[:,:26], 
                            vis[:,:26],
                            nw,
                            nh,
                            cx,
                            cy,
                            gt_seg,
                            pred_seg,
                            pck=pck2d,
                            mpjpe=mpjpe2d,
                            aligned=None,
                            type_step='test',
                            n_figs=1,
                            step=batch_idx,
                            align_root=True)
