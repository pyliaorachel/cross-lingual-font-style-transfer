import sys
import argparse

import torch.utils.data
import torchvision.datasets as datasets

from .net import StyleCNN
from ..utils.utils import *
from ..itn.utils import Logger



# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural style transfer.')
    parser.add_argument('--content', type=str,
                        help='Image to apply style on.')
    parser.add_argument('--style', type=str,
                        help='Image to extract style from.')
    parser.add_argument('--imsize', type=int,
                        help='Image size.')
    parser.add_argument('--output', type=str,
                        help='Path to save output image.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs. Default: 30')
    parser.add_argument('--log-epochs', type=int, default=10,
                        help='Number of epochs between training logs. Default: 10')

    args = parser.parse_args()
    return args

def isnan(x):
    return x != x

def train(style_image, content_image, imsize, epochs, output_path, log_epochs=10):
    # Load content and style images
    style = image_loader(style_image, imsize).type(dtype)
    content = image_loader(content_image, imsize).type(dtype)
    encoding = content_image.split('/')[-1].split('.')[1]
#     print(encoding)

    pastiche = image_loader(content_image, imsize).type(dtype)
    pastiche.data = torch.randn(pastiche.data.size()).type(dtype)

    # Train network
    style_cnn = StyleCNN(style, content, pastiche)
    
    logger = Logger(epochs, 1, log_per_iter=True)
    prev_loss = 1000

    for i in range(epochs):
        pastiche, content_loss, style_loss = style_cnn.train()
        pastiche.data.clamp_(0, 1)
        
        if isnan(content_loss) or isnan(style_loss):
            break
        
        if i % log_epochs == 0:
            print('Epoch: {}, content loss: {}, style loss: {}'.format(i, content_loss, style_loss))
            
        if (content_loss + style_loss) < prev_loss:
            prev_loss = (content_loss + style_loss)
            save_image(pastiche, (output_path + str(encoding) + '.png'), imsize)
            print('STYLE_LOSS_' + str(encoding) + ' : ' + str(style_loss))
        logger.log(losses={'content_loss':content_loss, 'style_loss':style_loss},  
                       images={'content':content, 'style':style, 'pastiche':pastiche})

    
    

if __name__ == '__main__':
    args = parse_args()

    train(args.style, args.content, args.imsize, args.epochs, args.output, args.log_epochs)
