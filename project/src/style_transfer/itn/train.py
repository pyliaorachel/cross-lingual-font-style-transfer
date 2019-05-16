import sys
import argparse

from torch.utils import data
import torchvision.datasets as datasets

from .net import ITN 
from ..utils.utils import *
from ..utils.dataset import Dataset


# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural style transfer.')
    parser.add_argument('--dataset', type=str,
                        help='Path to training images.')
    parser.add_argument('--style', type=str,
                        help='Image of target style.')
    parser.add_argument('--imsize', type=int,
                        help='Image size.')
    parser.add_argument('--output-model', type=str,
                        help='Path to save output model.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size. Default: 4')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs. Default: 30')
    parser.add_argument('--log-epochs', type=int, default=10,
                        help='Number of epochs between training logs. Default: 10')
    parser.add_argument('--save-epochs', type=int, default=10,
                        help='Number of epochs between saving models. Default: 10')

    args = parser.parse_args()
    return args

def train(dataset, style_image, imsize, epochs, batch_size, output_model_path, log_epochs=10, save_epochs=10):
    # Load training dataset
    train_set = Dataset(dataset, imsize)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Load target style image
    style = image_loader(style_image, imsize).type(dtype)

    # Train network
    itn = ITN(style)

    for i in range(epochs):
        for train_batch, _ in train_loader:
            pastiche, content_loss, style_loss = itn.train(train_batch)

            if i % log_epochs == 0:
                print('Epoch: {}, content loss: {}, style loss: {}'.format(i, content_loss, style_loss))

            if i % save_epochs == 0:
                itn.save(output_model_path, i)

if __name__ == '__main__':
    args = parse_args()

    train(args.dataset, args.style, args.imsize, args.epochs, args.batch_size, args.output_model, args.log_epochs, args.save_epochs)
