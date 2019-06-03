import argparse
import sys
import os

from torch.utils.data import DataLoader
import torch

from .utils import Logger
from ..utils.utils import *
from ..utils.dataset import Dataset
from .models import Generator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to test dataset.')
    parser.add_argument('--imsize', type=int, required=True,
                        help='Image size.')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name.')
    parser.add_argument('--epoch', type=int, default=-1,
                        help='The epoch the model stops training. Default: -1 (last epoch)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU computation.')
    parser.add_argument('--show', action='store_true',
                        help='To show with visdom. If not set, save to the predetermined path.')

    return parser.parse_args()

def evaluate(dataset, imsize, exp_name, epoch, cuda, show):

    if torch.cuda.is_available() and not cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda.')
    elif not torch.cuda.is_available() and cuda:
        print('WARNING: You do not have a CUDA device. Fallback to using CPU.')
    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor

    ###### Definition of variables ######
    # Networks
    netG_X2Y = Generator(device=device)
    netG_Y2X = Generator(device=device)

    # Load state dicts
    model_path = 'project/output/{}'.format(exp_name)
    if epoch == -1:
        G_X2Y_path = os.path.join(model_path, 'netG_X2Y.pth')
        G_Y2X_path = os.path.join(model_path, 'netG_Y2X.pth')
    else:
        G_X2Y_path = os.path.join(model_path, 'netG_X2Y_{}.pth'.format(epoch))
        G_Y2X_path = os.path.join(model_path, 'netG_Y2X_{}.pth'.format(epoch))
    netG_X2Y.load_state_dict(torch.load(G_X2Y_path))
    netG_Y2X.load_state_dict(torch.load(G_Y2X_path))
    netG_X2Y.to_device()
    netG_Y2X.to_device()

    # Set model's test mode
    netG_X2Y.eval()
    netG_Y2X.eval()

    # Dataset loader
    eval_set = Dataset(dataset, imsize, dtype=dtype)
    eval_loader = DataLoader(eval_set)

    ###### Testing######
    if not show:
        transfered_output_dir = 'project/output/{}/transfered'.format(exp_name)
        recovered_output_dir = 'project/output/{}/recovered'.format(exp_name)
        os.makedirs(transfered_output_dir, exist_ok=True)
        os.makedirs(recovered_output_dir, exist_ok=True)
    else:
        logger = Logger(1, len(eval_loader))

    for i, (real_X, fname) in enumerate(eval_loader):
        # Generate output
        fake_Y, _ = netG_X2Y(real_X)
        recovered_X, _ = netG_Y2X(fake_Y)

        transfered_fname = 'transfered_' + os.path.split(fname[0])[1]
        recovered_fname = 'recovered_' + os.path.split(fname[0])[1]
        if not show:
            # Rescale because output is in [-1, 1]
            fake_Y = 0.5 * (fake_Y.data + 1.0)
            recovered_X = 0.5 * (recovered_X.data + 1.0)

            # Save image files
            output_fname = os.path.join(transfered_output_dir, transfered_fname)
            save_image(fake_Y, output_fname, imsize)
            output_fname = os.path.join(recovered_output_dir, recovered_fname)
            save_image(recovered_X, output_fname, imsize)
        else:
            transfered_fname = os.path.splitext(transfered_fname)[0]
            recovered_fname = os.path.splitext(recovered_fname)[0]
            image_dict = {transfered_fname: fake_Y, recovered_fname: recovered_X}
            logger.log(images=image_dict)

        sys.stdout.write('\rGenerated images {} of {}'.format(i+1, len(eval_loader)))

    sys.stdout.write('\n')

if __name__ == '__main__':
    args = parse_args()

    evaluate(args.dataset, args.imsize, args.exp_name, args.epoch, args.cuda, args.show)
