'''
Modified from https://github.com/aitorzip/PyTorch-CycleGAN
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weights_init_normal, crop


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=8, device=torch.device('cpu')):
        super(Generator, self).__init__()

        # Initial convolution block       
        conv_blocks = [('refpad1', nn.ReflectionPad2d(3)),
                       ('conv1', nn.Conv2d(input_nc, 64, 7)),
                       ('norm1', nn.InstanceNorm2d(64)),
                       ('relu1', nn.ReLU(inplace=True)) ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for i in range(2):
            conv_blocks += [('conv' + str(i + 2), nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)),
                            ('norm' + str(i + 2), nn.InstanceNorm2d(out_features)),
                            ('relu' + str(i + 2), nn.ReLU(inplace=True)) ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        res_blocks = []
        for i in range(n_residual_blocks):
            res_blocks += [('resblk' + str(i + 1), ResidualBlock(in_features))]

        # Upsampling
        deconv_blocks = []
        out_features = in_features // 2
        for i in range(2):
            deconv_blocks += [('convt' + str(i + 1), nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)),
                              ('normt' + str(i + 1), nn.InstanceNorm2d(out_features)),
                              ('relut' + str(i + 1), nn.ReLU(inplace=True)) ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        output_blocks = [('outrefpad', nn.ReflectionPad2d(3)),
                         ('outconv', nn.Conv2d(64, output_nc, 7)),
                         ('outtanh', nn.Tanh()) ]

        self.conv = nn.Sequential(OrderedDict(conv_blocks))
        self.res = nn.Sequential(OrderedDict(res_blocks))
        self.deconv = nn.Sequential(OrderedDict(deconv_blocks))
        self.output = nn.Sequential(OrderedDict(output_blocks))
        self.apply(weights_init_normal)

        self.device = device

    def to_device(self):
        self.to(self.device)

    def release_device(self):
        self.cpu()

    def forward(self, x):
        z = self.conv(x)
        y = self.res(z)
        y = self.deconv(y)
        y = self.output(y)
        return y, z

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, device=torch.device('cpu')):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        self.apply(weights_init_normal)

        self.device = device

    def to_device(self):
        self.to(self.device)

    def release_device(self):
        self.cpu()

    def forward(self, x, crop_image=False, crop_type=None):
        if crop_image:
            x = crop(x, crop_type)
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
