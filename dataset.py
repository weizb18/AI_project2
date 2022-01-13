import numpy as np
import os
import h5py
import torch.utils.data as data
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, split="Training", transform = None):
        self.split = split
        self.transform = transform
        self.data = h5py.File('./data/data.h5', 'r', driver='core')
        if self.split == "Training":
            self.train_data = np.asarray(self.data['train_data']).reshape((28709,48,48))
            self.train_label = self.data['train_label']
        elif self.split == "Test":
            self.test_data = np.asarray(self.data['test_data']).reshape((7178,48,48))
            self.test_label = self.data['test_label']
    
    def __getitem__(self, index):
        if self.split == "Training":
            img, label = self.train_data[index], self.train_label[index]
        elif self.split == "Test":
            img, label = self.test_data[index], self.test_label[index]
        # RGB
        img = img[:,:,np.newaxis]
        img = np.concatenate((img,img,img),axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.split == "Training":
            return len(self.train_data)
        elif self.split == "Test":
            return len(self.test_data)
            