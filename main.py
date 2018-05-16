#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import Queue
import threading

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from msgList import MsgList
from flowlayout import FlowLayout

from speech import recog
from record import record

from Seq2Seq import chat
from emotion.detect import detect_emotion

image_queue = Queue.Queue()
running = True


HEAD1 = 'icons/head3.jpg'
HEAD2 = 'icons/head2.jpg'


def camera(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
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


class TextEdit(QTextEdit, QObject):
    '''支持ctrl+return信号发射的QTextEdit'''
    entered = Signal()

    def __init__(self, parent=None):
        super(TextEdit, self).__init__(parent)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Return:
            self.entered.emit()  # ctrl+enter 输入
        else:
            super(TextEdit, self).keyPressEvent(e)


class MsgInput(QWidget, QObject):
    '''自定义的内容输入控件，支持图像和文字的输入，文字输入按回车确认。'''
    textEntered = Signal(str)

    btnSize = 35
    teditHeight = 100

    def __init__(self, parent=None):
        super(MsgInput, self).__init__(parent)
        self.setContentsMargins(3, 3, 3, 3)

        self.textEdit = TextEdit()
        self.textEdit.setMaximumHeight(self.teditHeight)
        self.setMaximumHeight(self.teditHeight + self.btnSize)
        self.textEdit.setFont(QFont("Times", 15, QFont.Normal))
        self.textEdit.entered.connect(self.sendText)

        sendTxt = QPushButton(u'发送')
        sendTxt.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        sendTxt.setFixedHeight(self.btnSize)
        sendTxt.clicked.connect(self.sendText)

        sendSound = QPushButton(u'录音')
        sendSound.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        sendSound.setFixedHeight(self.btnSize)
        sendSound.clicked.connect(self.sendSound)

        hl = FlowLayout()
        hl.addWidget(sendTxt)
        hl.addWidget(sendSound)
        # hl.setMargin(0)

        vl = QVBoxLayout()
        vl.addWidget(self.textEdit)
        vl.addLayout(hl)
        # vl.setMargin(0)
        self.setLayout(vl)

    def sendText(self):
        txt = self.textEdit.toPlainText()
        if len(txt) > 0:
            self.textEntered.emit(txt)
            self.textEdit.clear()

    def sendSound(self):
        record()
        txt = recog().strip(u'，')
        if len(txt) > 0:
            self.textEntered.emit(txt)


class Backend(QThread):
    update = Signal(str)

    def __init__(self, txt):
        super(Backend, self).__init__()
        self.txt = txt

    def run(self):
        ans = chat(self.txt)
        ans = ans.decode('utf-8')
        self.update.emit(ans)


class PyqtChatApp(QSplitter):
    """聊天界面，QSplitter用于让界面可以鼠标拖动调节"""

    def __init__(self):
        super(PyqtChatApp, self).__init__(Qt.Horizontal)

        self.setWindowTitle('ChatBot')  # window标题
        self.setWindowIcon(QIcon('icons/chat.png'))  # ICON
        self.setMinimumSize(800, 600)  # 窗口最小大小

        self.msgList = MsgList()
        # self.msgList.setDisabled(True)  # 刚打开时没有聊天显示内容才对
        self.msgInput = MsgInput()
        self.msgInput.textEntered.connect(self.sendTextMsg)

        rSpliter = QSplitter(Qt.Vertical, self)
        self.msgList.setParent(rSpliter)
        self.msgInput.setParent(rSpliter)
        self.history = ['']
        # self.capture_thread = threading.Thread(
        #     target=camera, args=(0, image_queue, 320, 240, 10))
        # self.capture_thread.start()
        # self.camera_timer = QTimer()
        # self.camera_timer.timeout.connect(self.update_frame)
        # self.camera_timer.start(100)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.capture.set(cv2.CAP_PROP_FPS, 10)
        self.tmp_img = 'temp.png'

    def update_frame(self):
        if not image_queue.empty():
            frame = image_queue.get()
            img = frame['img']

            emotion = detect_emotion(img)
            print(emotion)
            cv2.putText(img, emotion, (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            # self.img_widget.setImage(img)

    @Slot(str)
    def sendTextMsg(self, txt):
        txt = unicode(txt)
        self.history.append(txt)
        # ques = ' '.join(self.history[-2:])
        ques = self.history[-1]
        self.msgList.addTextMsg(txt, False, HEAD1)
        self.backend = Backend(ques)
        self.backend.update.connect(self.sendAns)
        self.backend.start()

    @Slot(str)
    def sendAns(self, txt):
        txt = unicode(txt)
        self.msgList.addTextMsg(txt, True, HEAD2)
        ret, img = self.capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res, emotion = detect_emotion(img)
        cv2.imwrite(self.tmp_img, res)
        self.msgList.addTextMsg(emotion, True, HEAD2)
        self.msgList.addImageMsg(self.tmp_img, True, HEAD2)

    def keyPressEvent(self, e):
        if (e.key() == Qt.Key_Escape):
            sys.exit(app.exec_())
        super(PyqtChatApp, self).keyPressEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pchat = PyqtChatApp()
    pchat.show()
    sys.exit(app.exec_())
