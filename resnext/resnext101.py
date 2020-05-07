import resnext_101_32x4d_
import torch
from torch import nn
import torch.nn.functional as F

from config import *

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext101_32_path))

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]
        net.pop()
        self.view = net.pop()

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


class D4Net(nn.Module):
    def __init__(self, num_cls):
        super(D4Net, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext101_32_path))

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        net.pop()
        view = net.pop()
        self.classifier = nn.Sequential(view, nn.Linear(2048, num_cls))

    def forward(self, x1, x2):
        layer0_x1 = self.layer0(x1)
        layer1_x1 = self.layer1(layer0_x1)
        layer2_x1 = self.layer2(layer1_x1)
        layer3_x1 = self.layer3(layer2_x1)
        layer4_x1 = self.layer4(layer3_x1)

        layer0_x2 = self.layer0(x2)
        layer1_x2 = self.layer1(layer0_x2)
        layer2_x2 = self.layer2(layer1_x2)
        layer3_x2 = self.layer3(layer2_x2)
        layer4_x2 = self.layer4(layer3_x2)

        difference = layer4_x2 - layer4_x1
        difference = self.max_pool(difference)
        output = self.classifier(difference)

        return output, -self.max_pool(-F.cosine_similarity(layer4_x2, layer4_x1))
