import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class GramMatrix(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        features = x.view(N, C, H * W)
        G = torch.bmm(features, features.transpose(1, 2))

        return G.div(C * H * W)

class StyleCNN(object):
    def __init__(self, style, content, pastiche):
        super().__init__()

        self.style = style
        self.content = content
        self.pastiche = nn.Parameter(pastiche.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000000

        self.loss_network = models.vgg19(pretrained=True)

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.pastiche])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gram.to(self.device)
        self.loss_network.to(self.device)

    def train(self):
        content_losses = []
        style_losses = []
        def closure():
            self.optimizer.zero_grad()

            pastiche = self.pastiche.clone()
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()

            content_loss = 0
            style_loss = 0

            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                layer.to(self.device)

                pastiche, content, style = layer.forward(pastiche), layer.forward(content), layer.forward(style)

                if isinstance(layer, nn.Conv2d):
                    name = 'conv_' + str(i)

                    if name in self.content_layers:
                        # as suggested in medium post:
                        #content_loss += self.loss(pastiche * self.content_weight, content.detach() * self.content_weight)
                        # as in paper:
                        content_loss += self.content_weight * self.loss(pastiche, content.detach())

                    if name in self.style_layers:
                        pastiche_g, style_g = self.gram.forward(pastiche), self.gram.forward(style)
                        # as suggested in medium post:
                        #style_loss += self.loss(pastiche_g * self.style_weight, style_g.detach() * self.style_weight)
                        # as in paper
                        style_loss += self.style_weight * self.loss(pastiche_g, style_g.detach())

                if isinstance(layer, nn.ReLU):
                    i += 1

            total_loss = content_loss + style_loss
            total_loss.backward()

            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())

            return total_loss

        self.optimizer.step(closure)

        avg_content_loss = sum(content_losses) / len(content_losses)
        avg_style_loss = sum(style_losses) / len(style_losses)

        return self.pastiche, avg_content_loss, avg_style_loss
