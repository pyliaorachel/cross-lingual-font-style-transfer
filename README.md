# Cross-Lingual Font Style Transfer

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

## Image Transformation Network 
$ python -m project.src.style_transfer.itn.train --dataset PATH_TO_TRAIN_SET_DIR --style PATH_TO_STYLE_IMAGE --output-model PATH_TO_OUTPUT_MODEL --imsize IMAGE_SIZE --epochs EPOCHS --batch-size BATCH_SIZE --log-epochs LOG_EPOCHS --save-epochs SAVE_EPOCHS
$ python -m project.src.style_transfer.itn.eval --dataset PATH_TO_EVAL_SET_DIR --model PATH_TO_MODEL --imsize IMAGE_SIZE --output-dir OUTPUT_DIR

## CycleGAN 
### Remember to turn on visdom server first:
$ visdom # open http://localhost:8097/

### Then run:
$ python -m project.src.style_transfer.cyclegan.train \
    --content-dataset PATH_TO_CONTENT_TRAIN_SET_DIR --style-dataset PATH_TO_STYLE_TRAIN_SET_DIR \
    --imsize IMAGE_SIZE --exp-name EXPERIMENT_NAME \
    [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LEARNING_RATE] [--decay-epoch DECAY_EPOCH] [--cuda]
$ python -m project.src.style_transfer.cyclegan.eval \
    --dataset PATH_TO_EVAL_SET_DIR --imsize IMAGE_SIZE --exp-name EXPERIMENT_NAME [--cuda]
```

## Sample Output

##### Neural Style Transfer

###### Normal Image

<p align="center">
    <img src="img/neural_style_transfer/style.jpg?raw=true" width="128px" height="128px"/>
    <span> + </span>
    <img src="img/neural_style_transfer/content.jpg?raw=true" width="128px" height="128px"/>
    <span> = </span>
    <img src="img/neural_style_transfer/pastiche.png?raw=true" width="128px" height="128px"/>
</p>

## References

- [Neural Artistic Style Transfer: A Comprehensive Look](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199)
- [GitHub: aitorzip/PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN)
