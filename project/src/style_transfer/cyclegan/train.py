'''
Modified from https://github.com/aitorzip/PyTorch-CycleGAN
'''

import argparse
import itertools
import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

from .models import Generator, Discriminator
from .utils import ReplayBuffer, LambdaLR, Logger
from ..utils.utils import *
from ..utils.dataset import PairedDataset

CONTENT_LOSS = True
ID_LOSS = True
SGD = False
CROP_IMAGE = True
STYLE_AUGMENT = True
PATCH_GAN = False # unnecessary with CROP_IMAGE
SPECTRAL_NORM = False

GAN_LOSS_W = 1
CYCLE_LOSS_W = 10.0
ID_LOSS_W = 5.0
CONTENT_LOSS_W = 1
D_LR_W = 1
IMG_NC = 1

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-dataset', type=str, required=True,
                        help='Path to training content images.')
    parser.add_argument('--style-dataset', type=str, required=True,
                        help='Path to training style images.')
    parser.add_argument('--imsize', type=int, required=True,
                        help='Image size.')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs. Default: 200')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Yatch size. Default: 1')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate. Default: 0.0002')
    parser.add_argument('--decay-epoch', type=int, default=100,
                        help='Epoch to start linearly decaying the learning rate to 0. Default: 100')
    parser.add_argument('--d-steps', type=int, default=5,
                        help='Number of steps for training D. Default: 5.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU computation.')
    parser.add_argument('--save-model-epoch', type=int, default=10,
                        help='Epoch to save model. Default: 10')

    return parser.parse_args()

def save_models(netG_X2Y, netG_Y2X, netD_X, netD_Y, output_path, epoch=None):
    if epoch is None:
        torch.save(netG_X2Y.state_dict(), os.path.join(output_path, 'netG_X2Y.pth'))
        torch.save(netG_Y2X.state_dict(), os.path.join(output_path, 'netG_Y2X.pth'))
        torch.save(netD_X.state_dict(), os.path.join(output_path, 'netD_X.pth'))
        torch.save(netD_Y.state_dict(), os.path.join(output_path, 'netD_Y.pth'))
    else:
        torch.save(netG_X2Y.state_dict(), os.path.join(output_path, 'netG_X2Y_{}.pth'.format(epoch)))
        torch.save(netG_Y2X.state_dict(), os.path.join(output_path, 'netG_Y2X_{}.pth'.format(epoch)))
        torch.save(netD_X.state_dict(), os.path.join(output_path, 'netD_X_{}.pth'.format(epoch)))
        torch.save(netD_Y.state_dict(), os.path.join(output_path, 'netD_Y_{}.pth'.format(epoch)))

def train(content_dataset, style_dataset, imsize, exp_name, epochs, batch_size, lr, decay_epoch, d_steps, cuda, save_model_epoch):

    if torch.cuda.is_available() and not cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda.')
    elif not torch.cuda.is_available() and cuda:
        print('WARNING: You do not have a CUDA device. Fallback to using CPU.')
    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor
    byte_dtype = torch.cuda.ByteTensor if cuda and torch.cuda.is_available() else torch.ByteTensor

    ###### Definition of variables ######
    # Networks
    netG_X2Y = Generator(input_nc=IMG_NC, output_nc=IMG_NC, device=device)
    netG_Y2X = Generator(input_nc=IMG_NC, output_nc=IMG_NC, device=device)
    netD_X = Discriminator(input_nc=IMG_NC, patch_gan=PATCH_GAN, spec_norm=SPECTRAL_NORM, device=device)
    netD_Y = Discriminator(input_nc=IMG_NC, patch_gan=PATCH_GAN, spec_norm=SPECTRAL_NORM, device=device)

    netG_X2Y.to_device()
    netG_Y2X.to_device()
    netD_X.to_device()
    netD_Y.to_device()

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_content = torch.nn.MSELoss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_X2Y.parameters(), netG_Y2X.parameters()),
                                   lr=lr, betas=(0.5, 0.999))
    if SGD:
        optimizer_D_X = torch.optim.SGD(netD_X.parameters(), lr=lr*D_LR_W*0.1)
        optimizer_D_Y = torch.optim.SGD(netD_Y.parameters(), lr=lr*D_LR_W*0.1)
    else:
        optimizer_D_X = torch.optim.Adam(netD_X.parameters(), lr=lr*D_LR_W, betas=(0.5, 0.999))
        optimizer_D_Y = torch.optim.Adam(netD_Y.parameters(), lr=lr*D_LR_W, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)
    lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)
    lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)

    # Buffers initialization
    fake_X_buffer = ReplayBuffer()
    fake_Y_buffer = ReplayBuffer()

    # Dataset loader
    train_set = PairedDataset(content_dataset, style_dataset, imsize, dtype=dtype, input_nc=IMG_NC, augment=STYLE_AUGMENT)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Loss plot
    logger = Logger(epochs, len(train_loader), log_per_iter=True)

    ###### Training ######
    output_path = 'project/output/{}'.format(exp_name)
    os.makedirs(output_path, exist_ok=True)
    num_D_outputs = 1 if not PATCH_GAN else 14 * 14

    for epoch in range(epochs):
        for i, (real_X, real_Y) in enumerate(train_loader):
            if not PATCH_GAN:
                target_fake = torch.rand((len(real_X),)) * 0.5 + 0.7 # soft label: 0.7 - 1.2
                target_real = torch.rand((len(real_X),)) * 0.3       # soft label: 0 - 0.3
            else:
                target_fake = torch.rand((len(real_X), 14, 14)) * 0.5 + 0.7
                target_real = torch.rand((len(real_X), 14, 14)) * 0.3
            target_real = target_real.type(dtype)
            target_fake = target_fake.type(dtype)

            ###### Generators X2Y and Y2X ######
            optimizer_G.zero_grad()

            if ID_LOSS:
                # Identity loss
                # G_X2Y(Y) should equal Y if real Y is fed
                same_Y, _ = netG_X2Y(real_Y)
                loss_identity_Y = criterion_identity(same_Y, real_Y) * ID_LOSS_W

                # G_Y2X(X) should equal X if real X is fed
                same_X, _ = netG_Y2X(real_X)
                loss_identity_X = criterion_identity(same_X, real_X) * ID_LOSS_W

            # GAN loss
            fake_Y, content_real_X = netG_X2Y(real_X)
            pred_fake = netD_Y(fake_Y, crop_image=CROP_IMAGE, crop_type='random')
            loss_GAN_X2Y = criterion_GAN(pred_fake, target_real) * GAN_LOSS_W
            acc_GAN_X2Y = ((pred_fake < 0.5) == True).sum().item() / len(pred_fake) / num_D_outputs

            fake_X, content_real_Y = netG_Y2X(real_Y)
            pred_fake = netD_X(fake_X, crop_image=CROP_IMAGE, crop_type='center')
            loss_GAN_Y2X = criterion_GAN(pred_fake, target_real) * GAN_LOSS_W
            acc_GAN_Y2X = ((pred_fake < 0.5) == True).sum().item() / len(pred_fake) / num_D_outputs

            # Cycle loss
            recovered_X, content_fake_Y = netG_Y2X(fake_Y)
            loss_cycle_XYX = criterion_cycle(recovered_X, real_X) * CYCLE_LOSS_W

            recovered_Y, content_fake_X = netG_X2Y(fake_X)
            loss_cycle_YXY = criterion_cycle(recovered_Y, real_Y) * CYCLE_LOSS_W 

            # Content loss
            if CONTENT_LOSS:
                loss_content_X = criterion_content(content_fake_Y, content_real_X.detach()) * CONTENT_LOSS_W
                loss_content_Y = criterion_content(content_fake_X, content_real_Y.detach()) * CONTENT_LOSS_W

            # Total loss
            loss_G = loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_XYX + loss_cycle_YXY
            if ID_LOSS:
                loss_G += loss_identity_X + loss_identity_Y
            if CONTENT_LOSS:
                loss_G += loss_content_X + loss_content_Y
            loss_G.backward()

            optimizer_G.step()

            G_fake_X = fake_X
            G_fake_Y = fake_Y

            ###### Discriminator ######
            total_loss_D_X = 0
            total_loss_D_Y = 0
            total_acc_D_X = 0
            total_acc_D_Y = 0
            for d_step in range(d_steps):
                ###### Discriminator X ######
                optimizer_D_X.zero_grad()

                # Real loss
                pred_real = netD_X(real_X, crop_image=CROP_IMAGE, crop_type='random')
                loss_D_real = criterion_GAN(pred_real, target_real) * GAN_LOSS_W

                # Fake loss
                fake_X = fake_X_buffer.push_and_pop(fake_X)
                pred_fake = netD_X(fake_X.detach(), crop_image=CROP_IMAGE, crop_type='center')
                loss_D_fake = criterion_GAN(pred_fake, target_fake) * GAN_LOSS_W

                total_acc_D_X += (((pred_fake >= 0.5) == True).sum().item() + \
                                    ((pred_real < 0.5) == True).sum().item()) / (len(pred_real) + len(pred_fake)) / num_D_outputs

                # Total loss
                loss_D_X = (loss_D_real + loss_D_fake) * 0.5
                loss_D_X.backward()
                total_loss_D_X += loss_D_X

                optimizer_D_X.step()

                ###### Discriminator Y ######
                optimizer_D_Y.zero_grad()

                # Real loss
                pred_real = netD_Y(real_Y, crop_image=CROP_IMAGE, crop_type='center')
                loss_D_real = criterion_GAN(pred_real, target_real) * GAN_LOSS_W

                # Fake loss
                fake_Y = fake_Y_buffer.push_and_pop(fake_Y)
                pred_fake = netD_Y(fake_Y.detach(), crop_image=CROP_IMAGE, crop_type='random')
                loss_D_fake = criterion_GAN(pred_fake, target_fake) * GAN_LOSS_W

                total_acc_D_X += (((pred_fake >= 0.5) == True).sum().item() + \
                                    ((pred_real < 0.5) == True).sum().item()) / (len(pred_real) + len(pred_fake)) / num_D_outputs

                # Total loss
                loss_D_Y = (loss_D_real + loss_D_fake) * 0.5
                loss_D_Y.backward()
                total_loss_D_Y += loss_D_Y

                optimizer_D_Y.step()

            loss_D_X = total_loss_D_X / d_steps
            loss_D_Y = total_loss_D_Y / d_steps
            acc_D_X = total_acc_D_X / d_steps
            acc_D_Y = total_acc_D_Y / d_steps

            # Progress report (http://localhost:8097)
            loss_dict = {'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_X2Y + loss_GAN_Y2X),
                         'loss_G_cycle': (loss_cycle_XYX + loss_cycle_YXY), 'loss_D': (loss_D_X + loss_D_Y)}
            acc_dict = {'acc_G_GAN': (acc_GAN_X2Y + acc_GAN_Y2X) / 2, 'acc_D': (acc_D_X + acc_D_Y) / 2}
            image_dict = {'real_X': real_X, 'real_Y': real_Y, 'fake_X': G_fake_X, 'fake_Y': G_fake_Y,
                      'recovered_X': recovered_X, 'recovered_Y': recovered_Y}
            if ID_LOSS:
                loss_dict['loss_G_identity'] = (loss_identity_X + loss_identity_Y)
            if CONTENT_LOSS:
                loss_dict['loss_G_content'] = (loss_content_X + loss_content_Y)
            logger.log(loss_dict, acc_dict, images=image_dict)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_X.step()
        lr_scheduler_D_Y.step()

        # Save models checkpoints
        save_models(netG_X2Y, netG_Y2X, netD_X, netD_Y, output_path)

        if (epoch + 1) % save_model_epoch == 0:
            save_models(netG_X2Y, netG_Y2X, netD_X, netD_Y, output_path, epoch)

if __name__ == '__main__':
    args = parse_args()

    train(args.content_dataset, args.style_dataset, args.imsize, args.exp_name,
          args.epochs, args.batch_size, args.lr, args.decay_epoch, args.d_steps, args.cuda, args.save_model_epoch)
