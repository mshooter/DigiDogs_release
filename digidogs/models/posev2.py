"""
Copied from : https://github.com/facebookresearch/dinov2/issues/25#issuecomment-1585579842
https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/hubconf.py#L121
"""
import torch
import torch.nn as nn 
import kornia as K 

class LinearClassifierToken(nn.Module): 
    def __init__(self, in_channels, nc=1, tokenW=32, tokenH=32):
        super(LinearClassifierToken, self).__init__() 
        self.in_channels = in_channels
        self.W = tokenW
        self.H = tokenH
        self.nc = nc 
        self.conv = torch.nn.Conv2d(in_channels, nc, (1,1)) 

    def forward(self, x): 
        return self.conv(x.reshape(-1, self.H, self.W, self.in_channels).permute(0,3,1,2))

class LinearRegression(nn.Module): 
    def __init__(self, dim, num_labels=1): 
        super(LinearRegression, self).__init__() 
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels) 
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x): 
        return self.linear(x)

class SimplePose(nn.Module): 
    def __init__(self, n_keypoints=32, unfreeze=False, model_type='dino'):
        super().__init__()

        self.n_keypoints = n_keypoints
        self.unfreeze = unfreeze
        if model_type == 'dino':
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        else:
            raise NotImplementedError('there is not other model')

        self.linear_heatmapsxy = nn.Linear(768,512)
        self.linear_heatmapszy = nn.Linear(768,512)
        self.linear_heatmapsxz = nn.Linear(768,512)
        self.decoder_heatmapsxy = LinearClassifierToken(512, self.n_keypoints, 32, 32)
        self.decoder_heatmapszy = LinearClassifierToken(512, self.n_keypoints, 32, 32)
        self.decoder_heatmapsxz = LinearClassifierToken(512, self.n_keypoints, 32, 32)

        self.linear_segmentation = nn.Linear(768,512)
        self.decoder_segmentation = LinearClassifierToken(512, 1, 32,32)

        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            if self.unfreeze: 
                if ('blocks.9' in name) or ('blocks.10' in name) or ('blocks.11' in name) : 
                    param.requires_grad = True
        if self.unfreeze:
            self.encoder.blocks = self.encoder.blocks[:10] + [nn.Dropout(p=0.3)] +self.encoder.blocks[10:]

    def forward(self, x):
        """
        the input needs to be resized, normalized and interpolated to size 32 for masks
        """
        features = self.encoder.forward_features(x)
        #features = self.encoder.get_intermediate_layers(x, n=[0,3,6], return_class_token=True) # 4 layers do not require grad 

        cls_token = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # heatmaps
        h_outputxy = self.linear_heatmapsxy(patch_tokens)
        h_outputxy = self.decoder_heatmapsxy(h_outputxy)

        h_outputzy = self.linear_heatmapszy(patch_tokens)
        h_outputzy = h_outputzy.reshape(-1, 32, 32, 512).permute(0,3,1,2)
        size = int(h_outputzy.shape[-1])
        h_outputzy = torch.cat([t.permute(0,3,2,1) for t in h_outputzy.split(size,-3)],-3)
        h_outputzy = self.decoder_heatmapszy(h_outputzy)

        # === NEED TO CORRECT THIS, as this was wrong, but got ok results for WACV2024===
        h_outputxz = self.linear_heatmapsxz(patch_tokens)
        h_outputxz = h_outputxz.reshape(-1, 32, 32, 512).permute(0,3,1,2)
        size = int(h_outputxz.shape[-1])
        h_outputxz = torch.cat([t.permute(0,3,2,1) for t in h_outputxz.split(size,-3)],-3)
        h_outputxz = self.decoder_heatmapsxz(h_outputxz)

        h_outputxy = K.geometry.subpix.spatial_softmax2d(h_outputxy) 
        h_outputzy = K.geometry.subpix.spatial_softmax2d(h_outputzy) 
        h_outputxz = K.geometry.subpix.spatial_softmax2d(h_outputxz) 

        # segmentation
        seg_output = self.linear_segmentation(patch_tokens)
        seg_output = self.decoder_segmentation(seg_output) 

        return h_outputxy, h_outputzy, h_outputxz, seg_output 

if __name__ == "__main__":
    inp = torch.rand((4,3,448,448))
    model = SimplePose(unfreeze=True)
    hx, hy, hz,seg = model(inp)
    print(torch.cat((hx,hy,hz),dim=1).shape)
    print(hx.shape, hy.shape, hz.shape)
