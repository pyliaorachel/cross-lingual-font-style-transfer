import torch
import torchvision.transforms as transforms

from PIL import Image
import scipy.misc


def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    image = Image.open(image_name)
    image = loader(image)
    image = image.unsqueeze(0)
    return image

def save_image(input_t, path, imsize):
    unloader = transforms.ToPILImage()

    image = input_t.data.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    scipy.misc.imsave(path, image)
