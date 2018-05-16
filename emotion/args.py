# -*- coding: utf-8 -*-

import os
import time
import argparse


def prepare_dir(root, ds, trial_id=None):
    if not os.path.exists(root):
        os.mkdir(root)
    directory = os.path.join(root, ds.upper())
    if not os.path.exists(directory):
        os.mkdir(directory)
    if trial_id:
        directory = os.path.join(directory, str(trial_id))
        if not os.path.exists(directory):
            os.mkdir(directory)
    return directory


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--trainsize', type=int, default=None, help='training dataset size')
parser.add_argument('--valsize', type=int, default=None, help='validation dataset size')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--nocuda', dest='cuda', action='store_false')
parser.add_argument('--single', type=bool, default=True, help='single GPU')
parser.add_argument('--nosingle', dest='single', action='store_false')
parser.add_argument('--checkpoint', type=bool, default=False, help='use the best checkpoint or not')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping margin')
parser.add_argument('--num-classes', type=int, default='7', help='number of class')
parser.add_argument('--ds', type=str, default='ckplus', help='dataset name')
parser.add_argument('--eval-ds', type=str, default='ckplus', help='eval dataset name')
parser.add_argument('--tb-dir', type=str, default='logs', help='tensorboard logs dir')
parser.add_argument('--result-root', type=str, default='results', help='checkpoint save root')
parser.add_argument('--visual-root', type=str, default='visuals', help='visual result save root')
parser.add_argument('--trial-id', type=str, default='resnet50', help='trial id')
parser.add_argument('--eval-epoch', type=int, default=51, help='evaluate saved model selected by epoch number')
opt = parser.parse_args()


# 训练相关的超参数
ds = opt.ds
trial_id = opt.trial_id


# 训练日志信息
time_format = '%m-%d_%X'
current_time = time.strftime(time_format, time.localtime())
env_tag = '_'.join(map(str, [ds.upper(), trial_id, current_time, opt.batchsize, opt.lr]))
log_environment = os.path.join(opt.tb_dir, env_tag)   # tensorboard的记录环境


# 数据相关的参数
# 提供如下数据集：CK+
ckplus_rawimage_root = './datasets/CK+/Image_Emotion/'
ckplus_image_root = './datasets/CK+/images/'

dataset = {
    'ckplus': ckplus_image_root,
}

image_root = dataset[ds]
eval_image_root = dataset[opt.eval_ds]


# 结果评估相关的参数
result_root = opt.result_root
result_dir = prepare_dir(result_root, ds, trial_id)


with open(os.path.join(result_dir, 'args.txt'), 'w') as f:
    f.write(str(opt))

# checkpoint相关的参数
resnet_checkpoint = os.path.join(os.curdir, './models/resnet50-19c8e357.pth')  # 直接用pytorch训练的模型
# resnet_checkpoint = './models/resnet152-b121ed2d.pth'  # 直接用pytorch训练的模型

weight_pth_path = os.path.join(result_dir, ds + '_w.pth')
best_weight_pth_path = os.path.join(result_dir, ds + '_best_w.pth')
optimizer_pth_path = os.path.join(result_dir, ds + '_o.pth')
best_optimizer_pth_path = os.path.join(result_dir, ds + '_best_o.pth')


# 图示结果相关的超参数
visual_root = opt.visual_root
visual_dir = prepare_dir(visual_root, ds, trial_id)
