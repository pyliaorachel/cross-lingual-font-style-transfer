import sys
import os
import argparse

from torch.utils import data
import torchvision.datasets as datasets

from ..utils.utils import *
from ..utils.dataset import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from ..neural_style_transfer.net import GramMatrix

# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class StyleLoss(object):
    def __init__(self, style=None):
        super().__init__()
        self.style = style
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 100

        self.loss = nn.MSELoss()
        self.loss_network = models.vgg19(pretrained=True)
        self.gram = GramMatrix()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gram.to(self.device)
        self.loss_network.to(self.device)
    
    
    def eval(self, content, style):
        self.loss_network.eval()

        content = content.clone()
        pastiche = self.transform_network.forward(content)
        
        N = content.size()[0]

        style = self.style.clone()
        style = style.repeat(N, 1, 1, 1)
        # pastiche_img = pastiche
        
        # repeat
        pastiche = pastiche.repeat(1,3,1,1)
        content = content.repeat(1,3,1,1)
        style = style.repeat(1,3,1,1)

#         content_loss = 0
        style_loss = 0

        i = 1
        not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in list(self.loss_network.features):
            layer = not_inplace(layer)
            layer.to(self.device)

            pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)
#             print(pastiche.shape)
#             print(content.shape)
#             print(style.shape)

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

#                 if name in self.content_layers:
#                     content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)

                if name in self.style_layers:
                    pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                    style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

            if isinstance(layer, nn.ReLU):
                i += 1


        return style_loss
    
    def save(self, save_path, epoch):
        torch.save({
            'epoch': epoch,
            'loss_net_state_dict': self.loss_network.state_dict(),
            'transform_net_state_dict': self.transform_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.loss_network.load_state_dict(checkpoint['loss_net_state_dict'])
        self.transform_network.load_state_dict(checkpoint['transform_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return epoch

def parse_args():
    parser = argparse.ArgumentParser(description='Eval neural style transfer.')
    parser.add_argument('--dataset', type=str,
                        help='Path to testing images.')
    parser.add_argument('--style', type=str,
                        help='Image of target style.')
    parser.add_argument('--imsize', type=int,
                        help='Image size.')
   
   

    args = parser.parse_args()
    return args

def evaluate(dataset, style, imsize):
    # Load eval dataset
    eval_set = Dataset(dataset, imsize, dtype=dtype, input_nc=1)
    loader = data.DataLoader(eval_set)
    style = image_loader(style, imsize, input_nc=1).type(dtype)

    sloss = StyleLoss()
    
    style_loss = 0

    for content, fname in loader:
        style_loss += sloss.eval(content, style)
  
    print('style_loss: ', dataset, ' : ', style_loss / len(loader))
#         output_fname = os.path.join(output_dir, 'transfered_' + os.path.split(fname[0])[1])
#         save_image(pastiche, output_fname, imsize)
    return style_loss

if __name__ == '__main__':
    args = parse_args()

    evaluate(args.dataset, args.style, args.imsize)