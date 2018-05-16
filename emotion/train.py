# -*- coding: utf-8 -*-

'''
Training code
'''
from __future__ import print_function
from __future__ import absolute_import
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboard_logger import configure, log_value

from data import get_loader
from model import Model
from args import opt, log_environment, image_root, weight_pth_path
from utils import clip_gradient, adjust_lr


configure(log_environment, flush_secs=10)


# build models
model = Model(opt.num_classes)

if opt.cuda:
    train_loader = get_loader(image_root, batchsize=opt.batchsize, pin_memory=True)
    model.cuda()
else:
    train_loader = get_loader(image_root, batchsize=opt.batchsize, pin_memory=False)

total_step = len(train_loader)

params = model.parameters()
optimizer = torch.optim.Adam(params, lr=opt.lr)
crit = torch.nn.CrossEntropyLoss()

print("Let's go!")
for epoch in range(1, opt.epoch + 1):
    model.train()
    adjust_lr(optimizer, opt.lr, epoch)
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # Load data
        images, labels = pack
        images = Variable(images)
        labels = Variable(labels)
        if opt.cuda:
            images = images.cuda()
            labels = labels.cuda()
        # Forward
        scores = model(images)
        # Merge losses
        loss = crit(scores, labels)
        # Backward and update
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # Log
        _, preds = torch.max(scores, 1)
        num_correct = (preds == labels).sum()
        acc = num_correct.data[0] / float(len(labels))
        log_value('loss', loss.data[0], (epoch - 1) * total_step + i)
        log_value('acc', acc, (epoch - 1) * total_step + i)
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f Acc: %.2f' %
              (epoch, opt.epoch, i, total_step, loss.data[0], acc))
    save_path = weight_pth_path + '.%d' % epoch
    torch.save(model.state_dict(), save_path)
