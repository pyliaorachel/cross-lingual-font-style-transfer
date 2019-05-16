import os

import torch
import torchvision.transforms as transforms

from PIL import Image
import scipy.misc


def image_loader(image_name, imsize, first_dim=True):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    image = Image.open(image_name)
    image = loader(image)
    if first_dim:
        image = image.unsqueeze(0)
    return image

def save_image(input_t, path, imsize):
    unloader = transforms.ToPILImage()

    image = input_t.data.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    scipy.misc.imsave(path, image)

def save_batch_image(input_t, paths, imsize):
    N = input_t.size()[0]
    for input_tt, path in zip(input_t, paths):
        save_image(input_tt, path, imsize)

def is_image_file(fpath):
    if not os.path.isfile(fpath):
        return False
    fname, ext = os.path.splitext(fpath)
    return ext in ['.jpg', '.png']
