# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import time
import cv2
import numpy as np
import Queue
import threading
# import dlib
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax

from data import trans
from model import Model
from args import opt, weight_pth_path


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


running = False
capture_thread = None
image_queue = Queue.Queue()

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
idx_to_class = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy',
                5: 'sadness', 6: 'surprise'}

def camera(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    # capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    # capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    # capture.set(cv2.cv.CV_CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    idx = 0
    while running:
        frame = {}
        ret, img = capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame['img'] = img
        frame['idx'] = idx
        if queue.qsize() < 10:
            queue.put(frame)
            idx += 1


def openfile(filepath, queue):
    global running
    try:
        capture = cv2.VideoCapture(filepath)
    except Exception as e:
        print('Can not open %s. : %s' % (filepath, e))

    idx = 0
    while running:
        frame = {}
        ret, img = capture.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame['img'] = img
        frame['idx'] = idx
        queue.put(frame)
        idx += 1
    running = False


class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, width, height, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None
        self.window_width = width
        self.window_height = height

    def setImage(self, img):
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min([scale_w, scale_h])

        if scale == 0:
            scale = 1

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)

        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()    


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)

        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setGeometry(QtCore.QRect(0, 0, 640, 480))

        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.central_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.central_layout)

        self.vision_widget = QtWidgets.QWidget(self.central_widget)
        self.central_layout.addWidget(self.vision_widget)
        self.vision_layout = QtWidgets.QVBoxLayout(self.vision_widget)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.central_layout.addLayout(self.button_layout)

        self.start_button = QtWidgets.QPushButton()
        self.start_button.setText('start')
        self.start_button.setEnabled(True)
        self.button_layout.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton()
        self.stop_button.setText('stop')
        self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.stop_button)
        self.open_button = QtWidgets.QPushButton()
        self.open_button.setText('open')
        self.open_button.setEnabled(True)
        self.button_layout.addWidget(self.open_button)
        self.save_button = QtWidgets.QPushButton()
        self.save_button.setText('save')
        self.save_button.setEnabled(False)
        self.button_layout.addWidget(self.save_button)

        self.img_widget = OwnImageWidget(640, 480)
        self.vision_layout.addWidget(self.img_widget)

        self.start_button.clicked.connect(self.start_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.open_button.clicked.connect(self.open_clicked)

        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.camera_timer.start(100)

        self.main_layout.setStretch(20, 1)

        self.init_model()

    def init_model(self):
        self.model = Model(opt.num_classes)
        load_path = weight_pth_path + '.%d' % opt.eval_epoch
        if opt.cuda:
            weights = torch.load(load_path)
            self.model.cuda()
        else:
            weights = torch.load(load_path, map_location=lambda storage, loc: storage)
            self.model.cpu()
        self.model.load_state_dict(weights)
        self.model.eval()

    def start_clicked(self):
        global running
        global image_queue
        global score_queue
        running = True
        image_queue = Queue.Queue()
        score_queue = Queue.Queue()
        capture_thread = threading.Thread(
            target=camera, args=(0, image_queue, 320, 240, 10))
        capture_thread.start()
        self.start_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_clicked(self):
        global running
        running = False
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def open_clicked(self):
        global running
        global image_queue
        global score_queue
        running = True
        image_queue = Queue.Queue()
        score_queue = Queue.Queue()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(options=options)
        capture_thread = threading.Thread(target=openfile, args=(filepath, image_queue))
        capture_thread.start()
        # self.start_button.setEnabled(False)
        # self.open_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def update_frame(self):
        if not image_queue.empty():
            frame = image_queue.get()
            img = frame['img']
            idx = frame['idx']

            if idx % 50 != 0:
                self.img_widget.setImage(img)
                return
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(5, 5),
            )

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
                if opt.cuda:
                    face_var = face_var.cuda()
                score = self.model(face_var)
                score = softmax(score, 1)
                score, pred = torch.max(score, 1)
                emotion = '%s: %.2f' % (idx_to_class[pred.data[0].item()], score.data[0].item())
                # emotion = '%s: %.2f' % (idx_to_class[pred.data[0]], score.data[0])
                cv2.putText(img, emotion, (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                break

            self.img_widget.setImage(img)

    def closeEvent(self, event):
        global running
        running = False

    def keyPressEvent(self, e):
        k = e.key()

        if k == Qt.Key_Space:
            if self.start_button.isEnabled():
                self.start_clicked()
            else:
                self.stop_clicked()
        elif k == Qt.Key_Backspace:
            if self.windowState() & Qt.WindowFullScreen:
                self.showNormal()
            else:
                self.showFullScreen()
        elif k == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
        
