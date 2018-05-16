# -*- coding: utf-8 -*-

'''
Evaluation code
'''
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import torch
from torch.autograd import Variable

from data import get_loader
from model import Model
from args import opt, weight_pth_path, eval_image_root


def evaluate(model, eval_loader):
    total_step = len(eval_loader)

    num_test = 0
    num_correct = 0
    print("Let's go!")
    for i, pack in enumerate(eval_loader, start=1):
        # Load data
        images, labels = pack
        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if opt.cuda:
            images = images.cuda()
            labels = labels.cuda()
        # Forward
        t0 = time.time()
        scores = model(images)
        _, preds = torch.max(scores, 1)
        times = time.time()-t0
        bsz = len(labels)
        num_correct += (preds == labels).sum().data[0]
        num_test += bsz
        print('Step [%d/%d], FPS: %.2f' % (i, total_step, bsz / times))
    print('Accuracy: %.2f' % (num_correct / float(num_test)))

if __name__ == '__main__':
    # build models
    model = Model(opt.num_classes)
    load_path = weight_pth_path + '.%d' % opt.eval_epoch
    if opt.cuda:
        eval_loader = get_loader(eval_image_root, batchsize=opt.batchsize, pin_memory=True)
        weights = torch.load(load_path)
        model.cuda()
    else:
        eval_loader = get_loader(eval_image_root, batchsize=opt.batchsize, pin_memory=False)
        weights = torch.load(load_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(weights)
    model.eval()
    evaluate(model, eval_loader)
