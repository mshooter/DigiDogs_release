import torch
from torchvision import transforms
from torch.utils.data import DataLoader,ConcatDataset
import lightning.pytorch as pl
from lightning.pytorch.utilities import CombinedLoader
from digidogs.datasets.gtadogs_datasetv2 import GTADogs 
from digidogs.datasets.stanext import StanExt
from digidogs.datasets.rgbddog import RgbdDog
from digidogs.datasets.svm import Svm

class GtaModule(pl.LightningDataModule):
    def __init__(self, data_dir, anno_dir=None, data_dir1=None, train_type='gta', test_type='gta', batch_size=16, n_workers=12, test_dog=None, pin_memory=False):
        super().__init__()
        self.pin_memory = pin_memory
        self.test_dog = test_dog
        self.out_res = 32
        self.test_type = test_type
        self.train_type = train_type
        self.sigma=0.005
        self.anno_dir = anno_dir
        if self.train_type == 'both': 
            self.data_dir0 = data_dir 
            self.data_dir1 = data_dir1
        else:
            self.data_dir = data_dir
        self.batch_size = batch_size 
        self.n_workers = n_workers
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225) 
        self.img_size = 448
        self.transform_dino = transforms.Compose([
                                             transforms.ToPILImage(),
                                             transforms.Resize((self.img_size,self.img_size)),
                                             transforms.RandomApply([transforms.ColorJitter(brightness=0.5, hue=0.3)],p=0.45),  
                                             transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.35),     
                                             transforms.RandomGrayscale(p=0.5),                                  
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=self.mean, std=self.std)])
        self.transform_dinoog = transforms.Compose([
                                             transforms.ToPILImage(),
                                             transforms.Resize((self.img_size,self.img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=self.mean, std=self.std)])

    def setup(self, stage=None): 
        if self.train_type == 'svm': 
            dataset = Svm(data_dir=self.data_dir, transform=self.transform_dinoog, sigma=self.sigma, out_res=self.out_res)
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        if stage == 'fit' or stage is None: 
            # --- TRAIN SOLEY ON GTA DATASET ---
            if self.train_type == 'gta': 
                self.train_dataset = GTADogs(self.data_dir, 
                                             self.anno_dir, 
                                             data_type = 'train', 
                                             sigma = self.sigma, 
                                             transform = self.transform_dino, 
                                             out_res = self.out_res)

                self.val_dataset = GTADogs(self.data_dir, 
                                           self.anno_dir, 
                                           data_type='val', 
                                           sigma=self.sigma, 
                                           transform=self.transform_dinoog, 
                                           out_res = self.out_res) # what if we included a bit of stanford extra in there?

            # --- TRAINING SOLEY WITH STANFORD EXTRA ---
            elif self.train_type == 'stanford': 
                self.train_dataset = StanExt(data_dir=self.data_dir, 
                                    data_type='train', 
                                    transform=self.transform_dino,
                                    out_res=self.out_res,
                                    sigma=self.sigma)

                self.val_dataset = StanExt(data_dir=self.data_dir, 
                                    data_type='val', 
                                    transform=self.transform_dino,
                                    out_res=self.out_res,
                                    sigma=self.sigma)

            # --- TRAINING WITH BOTH DATASETS ---
            elif self.train_type == 'both':
                # === stanford === 
                self.train_stanext = StanExt(data_dir=self.data_dir1, 
                                    data_type='train', 
                                    transform=self.transform_dino,
                                    out_res=self.out_res,
                                    sigma=self.sigma)

                self.val_stanext = StanExt(data_dir=self.data_dir1, 
                                    data_type='val', 
                                    transform=self.transform_dino,
                                    out_res=self.out_res,
                                    sigma=self.sigma)

                # === gta === 
                self.train_gta = GTADogs(self.data_dir0, 
                                    data_type='train', 
                                    sigma=self.sigma, 
                                    transform=self.transform_dino, 
                                    out_res=self.out_res)
                self.val_gta = GTADogs(self.data_dir0, 
                                  data_type='val', 
                                  sigma=self.sigma, 
                                  transform=self.transform_dino, 
                                  out_res=self.out_res)
        
                # === combine ===
                #self.train_dataset = ConcatDataset([train_stanext, train_gta]) 
                #self.val_dataset = ConcatDataset([val_stanext, val_gta])
                #if self.is_val3d: 
                #    self.val_dataset = self.val_gta
                #else:
                #    self.val_dataset = self.val_stanext
            elif self.train_type == "rgbd":
                dataset = RgbdDog(train_type="train", transform=self.transform_dino)
                train_size = int(0.8 * len(dataset))
                test_size = len(dataset) - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
                

        if stage == 'test' or stage is None: 
            # test on gta 
            if self.test_type == 'gta':
                self.test_dataset = GTADogs(self.data_dir, 
                                            self.anno_dir,
                                            data_type='test', 
                                            sigma=self.sigma, 
                                            transform=self.transform_dinoog, 
                                            out_res=self.out_res)

            # test on stanford
            elif self.test_type == 'stanford':
                self.test_dataset = StanExt(data_dir="/vol/research/animal_motion_data/datasets_3rd_party/Stanford_Dogs", 
                                    data_type='test', 
                                    transform=self.transform_dinoog,
                                    out_res=self.out_res, 
                                    sigma=self.sigma)

            elif self.test_type == 'rgbd':
                self.test_dataset = RgbdDog(data_dir=self.data_dir, train_type="test", test_dog=self.test_dog, transform=self.transform_dinoog)

    def train_dataloader(self):
        if self.train_type == "both":
            loader_gta = DataLoader(self.train_gta, 
                    batch_size=self.batch_size, 
                    num_workers=self.n_workers//2, 
                    shuffle=True, 
                    drop_last=True)
            loader_stan = DataLoader(self.train_stanext, 
                    batch_size=self.batch_size, 
                    num_workers=self.n_workers//2, 
                    shuffle=True, 
                    drop_last=True)
            loaders = {'gta': loader_gta, 'stanford':loader_stan}
            combined_loader = CombinedLoader(loaders)
            return loaders
        else:
            return DataLoader(self.train_dataset, 
                    batch_size=self.batch_size, 
                    num_workers=self.n_workers, 
                    shuffle=True, 
                    drop_last=True,
                    pin_memory=self.pin_memory)
                    
    
    def val_dataloader(self):
        if self.train_type == "both":
            loader_gta = DataLoader(self.val_gta, 
                    batch_size=self.batch_size, 
                    num_workers=self.n_workers//2, 
                    shuffle=False, 
                    drop_last=True)
            loader_stan = DataLoader(self.val_stanext, 
                    batch_size=self.batch_size, 
                    num_workers=self.n_workers//2, 
                    shuffle=False, 
                    drop_last=True)
            loaders = {'gta': loader_gta, 'stanford':loader_stan}
            combined_loader = CombinedLoader(loaders)
            return loaders
        else:
            return DataLoader(self.val_dataset, 
                    batch_size=self.batch_size, 
                    num_workers=self.n_workers, 
                    shuffle=False, 
                    drop_last=True,
                    pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.n_workers, 
                shuffle=False, 
                pin_memory=self.pin_memory,
                drop_last=True)

