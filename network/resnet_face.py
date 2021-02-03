import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import numpy as np
import os
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from .getpretrained import get_model_files


class ResNet_face(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.backbone == 'resnet18':
            self.backbone = models.resnet18()
            last_channels = 512
        elif args.backbone == 'resnet34':
            self.backbone = models.resnet34()
            last_channels = 512
        elif args.backbone == 'resnet50':
            self.backbone = models.resnet50()
            last_channels = 2048
        elif args.backbone == 'resnet101':
            self.backbone = models.resnet101()
            last_channels = 2048
        elif args.backbone == 'resnet152':
            self.backbone = models.resnet152()
            last_channels = 2048
        else:
            logging.info(f'No such network {args.network}')
            os._exit(0)
        
        if args.load:
                loadfilename = get_model_files(args.backbone)
                self.backbone.load_state_dict(torch.load(loadfilename))
                logging.info(f'Model loaded from {loadfilename}')

        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4)

        self.bn1 = nn.BatchNorm2d(last_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(8*8*last_channels, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)

        return output
