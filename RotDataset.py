import numpy as np 
from PIL import Image

import torch 
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

class RotDataset(Dataset):
    def __init__(self, dataset, train_mode=True):
        self.dataset = dataset 
        self.num_data = len(dataset)
        self.train_mode = train_mode
       
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.random_crop = transforms.RandomResizedCrop(size=32, scale=(0.2, 1.))

    def __getitem__(self, index):
        x_orig, classifier_target = self.dataset[index]

        x_orig =  np.copy(x_orig)

        if self.train_mode == True:
            x_orig = Image.fromarray(x_orig)
            x_orig = self.random_crop(x_orig)
            x_orig = np.asarray(x_orig)
        
        x_tf_0 = self.normalize(functional.to_tensor(np.copy(x_orig)))
        x_tf_90 = self.normalize(functional.to_tensor(np.rot90(x_orig.copy(), k=1).copy()))
        x_tf_180 = self.normalize(functional.to_tensor(np.rot90(x_orig.copy(), k=2).copy()))
        x_tf_270 = self.normalize(functional.to_tensor(np.rot90(x_orig.copy(), k=3).copy()))
        
        return x_tf_0, x_tf_90, x_tf_180, x_tf_270, classifier_target

    def __len__(self):
        return self.num_data
