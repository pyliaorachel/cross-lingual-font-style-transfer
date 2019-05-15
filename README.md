# Cross-lingual Font Style Transfer

Cross-lingual font style transfer from English fonts to Chinese characters.

## Methods

- Neural Style Transfer
    - [A Neural Algorithm of Artistic Style, Gatys et al.](https://arxiv.org/pdf/1508.06576v2.pdf)
- Image Transformation Network
    - [Perceptual Losses for Real-Time Style Transfer and Super-Resolution, Johnson et al.](https://arxiv.org/pdf/1603.08155.pdf)
- CycleGAN
    - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, Zhu et al.](https://arxiv.org/pdf/1703.10593.pdf)

## Usage 

```bash
# Create virtual environment
$ conda env create -f environment.yml
$ conda activate style_transfer

# Run
## Neural style transfer
$ python -m project.src.style_transfer.neural_style_transfer.train --content PATH_TO_CONTENT_IMAGE --style PATH_TO_STYLE_IMAGE --output PATH_TO_OUTPUT_IMAGE --imsize IMAGE_SIZE --epochs EPOCHS --log-epochs LOG_EPOCHS
```
