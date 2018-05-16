# -*- coding: utf-8 -*-


# 引入Speech SDK
from aip import AipSpeech

# 定义常量
APP_ID = '10620591'  # '你的 App ID'
API_KEY = 'BYeprilBjMxp8jaNhfw2Rb7Z'  # '你的 API Key'
SECRET_KEY = '27e5ca5a4142cbc1edfe782ee03040e5'  # '你的 Secret Key'


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def recog():
    # 初始化AipSpeech对象
    aipSpeech = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    # 识别本地文件
    res = aipSpeech.asr(get_file_content('output.wav'), 'wav', 16000, {
        'lan': 'zh',
    })
    print res['result'][0]
    return res['result'][0]


if __name__ == '__main__':
    recog()
