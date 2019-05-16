import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from ..neural_style_transfer.net import GramMatrix


class ITN(object):
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
        self.transform_network = nn.Sequential(
            nn.ReflectionPad2d(40),
            nn.Conv2d(3, 32, 9, stride=1, padding=4),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(32, 3, 9, stride=1, padding=4),
        )
        self.optimizer = optim.Adam(self.transform_network.parameters(), lr=1e-3)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gram.to(self.device)
        self.loss_network.to(self.device)
        self.transform_network.to(self.device)

    def train(self, content):
        self.loss_network.train()
        self.transform_network.train()

        self.optimizer.zero_grad()

        N = content.size()[0]

        content = content.clone()
        style = self.style.clone()
        style = style.repeat(N, 1, 1, 1)
        pastiche = self.transform_network.forward(content)

        content_loss = 0
        style_loss = 0

        i = 1
        not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in list(self.loss_network.features):
            layer = not_inplace(layer)
            layer.to(self.device)

            pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in self.content_layers:
                    content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)

                if name in self.style_layers:
                    pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                    style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)

            if isinstance(layer, nn.ReLU):
                i += 1

        total_loss = content_loss + style_loss
        total_loss.backward()

        self.optimizer.step()

        return pastiche, content_loss.item(), style_loss.item()
    
    def eval(self, content):
        self.loss_network.eval()
        self.transform_network.eval()

        content = content.clone()
        pastiche = self.transform_network.forward(content)

        return pastiche

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
