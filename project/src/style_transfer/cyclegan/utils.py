import random
import time
import datetime
import sys

import torch
from visdom import Visdom
import numpy as np

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, log_per_iter=False):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.iter = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows_batch = {}
        self.loss_windows = {}
        self.accs = {}
        self.acc_windows_batch = {}
        self.acc_windows = {}
        self.image_windows = {}
        self.log_per_iter = log_per_iter

    def log(self, losses=None, accs=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        if losses is not None:
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name].item()
                else:
                    self.losses[loss_name] += losses[loss_name].item()

                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        if accs is not None:
            for i, acc_name in enumerate(accs.keys()):
                if acc_name not in self.accs:
                    self.accs[acc_name] = accs[acc_name]
                else:
                    self.accs[acc_name] += accs[acc_name]

                if (i+1) == len(accs.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (acc_name, self.accs[acc_name]/self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (acc_name, self.accs[acc_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        if images is not None:
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
                else:
                    self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            if losses is not None:
                for loss_name, loss in self.losses.items():
                    if loss_name not in self.loss_windows_batch:
                        self.loss_windows_batch[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                           opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                    else:
                        self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows_batch[loss_name], update='append')
                    # Reset losses for next epoch
                    self.losses[loss_name] = 0.0

            # Plot accuracies
            if accs is not None:
                for acc_name, acc in self.accs.items():
                    if acc_name not in self.acc_windows_batch:
                        self.acc_windows_batch[acc_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([acc/self.batch]),
                                                                           opts={'xlabel': 'epochs', 'ylabel': acc_name, 'title': acc_name})
                    else:
                        self.viz.line(X=np.array([self.epoch]), Y=np.array([acc/self.batch]), win=self.acc_windows_batch[acc_name], update='append')
                    # Reset accs for next epoch
                    self.accs[acc_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        if self.log_per_iter:
            # End of iteration
            if losses is not None:
                for loss_name, loss in losses.items():
                    loss = loss.item()
                    if loss_name not in self.loss_windows:
                        self.loss_windows[loss_name] = self.viz.line(X=np.array([self.iter]), Y=np.array([loss]),
                                                                     opts={'xlabel': 'iterations', 'ylabel': loss_name, 'title': loss_name})
                    else:
                        self.viz.line(X=np.array([self.iter]), Y=np.array([loss]), win=self.loss_windows[loss_name], update='append')

            if accs is not None:
                for acc_name, acc in accs.items():
                    if acc_name not in self.acc_windows:
                        self.acc_windows[acc_name] = self.viz.line(X=np.array([self.iter]), Y=np.array([acc]),
                                                                     opts={'xlabel': 'iterations', 'ylabel': acc_name, 'title': acc_name})
                    else:
                        self.viz.line(X=np.array([self.iter]), Y=np.array([acc]), win=self.acc_windows[acc_name], update='append')

            self.iter += 1
            sys.stdout.write('\n')

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def center_crop(t_img, ratio=0.25):
    if ratio >= 1:
        return t_img
    h, w = t_img.size()[-2:]
    crop_h, crop_w = h * ratio, w * ratio
    start_h, end_h, start_w, end_w = int(h // 2 - crop_h // 2), int(h // 2 + crop_h // 2), int(w // 2 - crop_w // 2), int(w // 2 + crop_w // 2)

    if len(t_img.size()) == 3:
        return t_img[:, start_h:end_h, start_w:end_w]
    else:
        return t_img[:, :, start_h:end_h, start_w:end_w]

def random_crop(t_img, ratio=0.25):
    if ratio >= 1:
        return t_img
    h, w = t_img.size()[-2:]
    crop_h, crop_w = h * ratio, w * ratio

    start_h, start_w = int((h - crop_h) * random.random()), int((w - crop_w) * random.random())
    end_h, end_w = int(start_h + crop_h), int(start_w + crop_w)

    if len(t_img.size()) == 3:
        return t_img[:, start_h:end_h, start_w:end_w]
    else:
        return t_img[:, :, start_h:end_h, start_w:end_w]

def crop(t_img, crop_type=None, ratio=0.25):
    if crop_type == 'center':
        return center_crop(t_img, ratio=ratio)
    elif crop_type == 'random':
        return random_crop(t_img, ratio=ratio)
    return t_img
