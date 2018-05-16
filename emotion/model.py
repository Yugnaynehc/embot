# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from args import resnet_checkpoint
from args import opt


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.resnet = models.resnet50()
        if opt.cuda:
            pmodel = torch.load(resnet_checkpoint)
        else:
            pmodel = torch.load(resnet_checkpoint, map_location=lambda storage, location: storage)
        self.resnet.load_state_dict(pmodel)
        del self.resnet.fc
        expansion = 4
        self.fc = nn.Linear(512 * expansion, num_classes)

    def forward(self, images):
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)  # 64 x 56 x 56
        x = self.resnet.layer2(x)  # 128 x 28 x 28
        x = self.resnet.layer3(x)  # 256 x 14 x 14
        x = self.resnet.layer4(x)  # 512 x 7 x 7

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
