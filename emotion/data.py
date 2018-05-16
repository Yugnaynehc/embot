# -*- coding: utf-8 -*-

'''
Preapre salient object segmentation data
'''

import os
import torch.utils.data as data
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.datasets import ImageFolder

from args import image_root
from args import ckplus_rawimage_root, ckplus_image_root


trans = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_loader(image_root, batchsize=10, size=None, shuffle=True, num_workers=3, pin_memory=False):
    dataset = ImageFolder(image_root, transform=trans)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def prepare_ckplus(raw_root, target_root):
    import cv2
    import dlib
    detector = dlib.get_frontal_face_detector()

    if not os.path.exists(target_root):
        os.mkdir(target_root)
    for root, _, fnames in os.walk(raw_root):
        if not fnames:
            continue
        label = (root.split('/')[-1]).split('_')[-1]
        target_dir = os.path.join(target_root, label)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for fname in fnames:
            filepath = os.path.join(root, fname)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            dets = detector(img, 1)
            for i, face in enumerate(dets):
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                face_img = img[top:bottom, left:right]
                face_img = cv2.resize(face_img, (224, 224))
                target_path = os.path.join(target_dir, '%d_%s' % (i, fname))
                cv2.imwrite(target_path, face_img)


if __name__ == '__main__':
    # prepare_ckplus(ckplus_rawimage_root, ckplus_image_root)
    train_loader = get_loader(image_root)
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {v:k for (k, v) in class_to_idx.items()}
    print(class_to_idx)
    print(idx_to_class)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size(), d[1].size())
    print(d[1])
