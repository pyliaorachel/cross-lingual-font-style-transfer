import sys
import os
import argparse

from torch.utils import data
import torchvision.datasets as datasets

from .net import ITN 
from ..utils.utils import *
from ..utils.dataset import Dataset


# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser(description='Eval neural style transfer.')
    parser.add_argument('--dataset', type=str,
                        help='Path to testing images.')
    parser.add_argument('--imsize', type=int,
                        help='Image size.')
    parser.add_argument('--model', type=str,
                        help='Path to saved model.')
    parser.add_argument('--output-dir', type=str,
                        help='Path to output directory.')

    args = parser.parse_args()
    return args

def evaluate(dataset, imsize, model_path, output_dir):
    # Load eval dataset
    eval_set = Dataset(dataset, imsize, dtype=dtype)
    loader = data.DataLoader(eval_set)

    # Load network
    itn = ITN()
    itn.load(model_path)

    for content, fname in loader:
        pastiche = itn.eval(content)

        output_fname = os.path.join(output_dir, 'transfered_' + os.path.split(fname[0])[1])
        save_image(pastiche, output_fname, imsize)

if __name__ == '__main__':
    args = parse_args()

    evaluate(args.dataset, args.imsize, args.model, args.output_dir)
