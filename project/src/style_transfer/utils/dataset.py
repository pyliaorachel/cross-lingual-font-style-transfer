import os
import random

import torch
from torch.utils import data

from .utils import image_loader, is_image_file


class Dataset(data.Dataset):
    def __init__(self, dataset, imsize, dtype=torch.FloatTensor, input_nc=3):
        self.dataset = dataset
        self.imsize = imsize
        self.dtype = dtype
        self.input_nc = input_nc

        self.list_IDs = [os.path.join(dataset, fname) for fname in os.listdir(dataset) if is_image_file(os.path.join(dataset, fname))]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        image_name = self.list_IDs[index]
        X = image_loader(image_name, self.imsize, first_dim=False, input_nc=self.input_nc).type(self.dtype)
        return X, image_name

class PairedDataset(data.Dataset):
    def __init__(self, dataset, paired_dataset, imsize, dtype=torch.FloatTensor, input_nc=3):
        self.dataset = dataset
        self.paired_dataset = paired_dataset
        self.imsize = imsize
        self.dtype = dtype
        self.input_nc = input_nc

        self.list_IDs = self.get_list_IDs(dataset)
        self.paired_list_IDs = self.get_list_IDs(paired_dataset)
        self.n_paired = len(self.paired_list_IDs)

    def get_list_IDs(self, dataset):
        return [os.path.join(dataset, fname) for fname in os.listdir(dataset) if is_image_file(os.path.join(dataset, fname))]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        image_name = self.list_IDs[index]
        X = image_loader(image_name, self.imsize, first_dim=False, input_nc=self.input_nc).type(self.dtype)

        paired_image_name = self.paired_list_IDs[random.randint(0, self.n_paired - 1)]
        Y = image_loader(paired_image_name, self.imsize, first_dim=False, input_nc=self.input_nc).type(self.dtype)

        return X, Y
