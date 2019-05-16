import torch
import os

from torch.utils import data

from .utils import image_loader, is_image_file


class Dataset(data.Dataset):
    def __init__(self, dataset, imsize):
        self.dataset = dataset
        self.imsize = imsize
        self.list_IDs = [os.path.join(dataset, fname) for fname in os.listdir(dataset) if is_image_file(os.path.join(dataset, fname))]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        image_name = self.list_IDs[index]
        X = image_loader(image_name, self.imsize, first_dim=False)
        return X, image_name
