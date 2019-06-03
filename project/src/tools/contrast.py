import argparse
import os

from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset to process.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset
    output_dir = os.path.join(dataset, 'contrast')
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(dataset):
        fpath = os.path.join(dataset, fname)
        if os.path.isfile(fpath) and (fname.endswith('.jpg') or fname.endswith('.png')):
            output_fpath = os.path.join(output_dir, fname)

            image = Image.open(fpath)
            image_array = np.array(image)
            image_array[image_array < 255] = 0
            image_processed = Image.fromarray(np.uint8(image_array))
            image_processed.save(output_fpath)
