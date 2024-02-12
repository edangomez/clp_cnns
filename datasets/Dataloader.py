import glob
import torch
from torch.utils.data import Dataset
import cv2 as cv
from typing import Callable, Optional
import pandas as pd
import os


class Caltech101(Dataset):
    def __init__(
        self,
        gt_path,
        imgs_source,
        transformations: Optional[Callable] = None
    ):
        self._dataset = pd.read_csv(gt_path)
        self.ds_dic = self._dataset.to_dict(orient='records')
        self.transform = transformations
        self.labels = []
        self.image_paths = []
        
        for example in self.ds_dic:
            self.labels.append(example['label_idx'])
            self.image_paths.append(os.path.join(imgs_source, example['path']))
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_paths)
    
    def _load_image(self, idx: int):
        # TODO: load image
        path = self.image_paths[idx]
        image = cv.imread(path, cv.IMREAD_COLOR)
        image = cv.resize(image, (227, 227), interpolation=cv.INTER_CUBIC)
        
        return image
    
    def _load_label(self, idx):
        # TODO: load ground truth
        label = self.labels[idx]
        
        return label
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        
        label = self._load_label(idx)
        image = self._load_image(idx)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label