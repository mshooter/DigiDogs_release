import os
import glob
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class DemoDataset(Dataset): 
    def __init__(self, data_folder, transform=None):
        
        images = glob.glob(os.path.join(data_folder ,"*.png"))
        self.images = images + glob.glob(os.path.join(data_folder ,"*.jpg"))
        self.transform = transform 
    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, index): 
        img_pth = self.images[index]
        img = np.array(Image.open(img_pth).convert('RGB')) 
        h, w, _ = img.shape
        cx = w/2
        cy = h/2
        
        if self.transform is not None:
            img = self.transform(img)

        target = {'cx': cy, 'cy': cy, 'nw':w, 'nh':h, "img_pth":img_pth}
        return img, target 
