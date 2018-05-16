
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax

from .data import trans
from .model import Model
from .args import opt, weight_pth_path


face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
idx_to_class = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy',
                5: 'sadness', 6: 'surprise'}


model = Model(opt.num_classes)
load_path = weight_pth_path + '.%d' % opt.eval_epoch
weights = torch.load(load_path, map_location=lambda storage, loc: storage)
model.cpu()
model.load_state_dict(weights)
model.eval()


def detect_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
    )

    emotion = ''
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 2)
        # face_img = img[y:y+h, x:x+w]
        face_img = gray[y:y+h, x:x+w]
        face_img = np.tile(face_img[:, :, None], 3)
        face_img = cv2.resize(face_img, (224, 224))
        # cv2.imshow('a', face_img)
        # cv2.waitKey(0)
        face_tensor = trans(face_img).unsqueeze(0)
        face_var = Variable(face_tensor, volatile=True)
        score = model(face_var)
        score = softmax(score, 1)
        score, pred = torch.max(score, 1)
        emotion = '%s: %.2f' % (idx_to_class[pred.data[0].item()], score.data[0].item())
        cv2.putText(img, emotion, (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        break

    return img, emotion
