#coding=utf-8
import gensim
import logging
import multiprocessing
import os
import re
import sys

import jieba
import codecs
import numpy as np
import six.moves.cPickle as pickle
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def clean_face(raw_):
    cleanr = re.compile(r'\[.*\]|\{.*\}|\(.*\)|\<.*\>',re.S)
    cleantext = re.sub(cleanr, '', raw_)
    return cleantext

def clean_notchinese(raw_):
    cleanr = re.compile(ur"[^\u4e00-\u9fa5]+",re.S)
    #print raw_
    cleantext = re.sub(cleanr, '', raw_.decode('utf8',errors='ignore'))
    return cleantext.encode('utf8',errors='ignore')



def Init_Sentences(dirname,dict):
    Qs=[]
    As=[]

    for root, dirs, files in os.walk(dirname):
        for filename in files:
            file_path = root + '/' + filename
            #temp_file= codecs.open(file_path,"r", encoding="utf8",errors='ignore')
            temp_file = open(file_path, "r")
            print temp_file

            preline=""

            while True:
                line1 = temp_file.readline()

                if line1:

                    sline0 = preline.strip()
                    rline0 = clean_face(sline0)
                    rline0 = clean_notchinese(rline0)

                    if rline0 == "":
                        preline = line1
                        continue


                    sline1 = line1.strip()
                    rline1 = clean_face(sline1)
                    rline1 = clean_notchinese(rline1)

                    if rline1 == "":
                        preline = line1
                        continue

                    seg_list0 = jieba.cut(rline0)
                    seg_list1 = jieba.cut(rline1)

                    QQ = [w for w in seg_list0]
                    AA = [w for w in seg_list1]

                    temp_res_Q = [dict[w].index for w in QQ if w in dict]
                    temp_res_A = [dict[w].index for w in AA if w in dict]

                    Qs.append(temp_res_Q)
                    As.append(temp_res_A)

                    preline = line1


                else:
                    break
    with open('./Sentences_Qs_100.pkl', 'wb') as f:
        pickle.dump(Qs, f,-1)

    with open('./Sentences_As_100.pkl', 'wb') as f:
        pickle.dump(As, f,-1)

    return


def Split_Sentences(filename, n=3):

    begin = time()
    print "Loading data --------"
    Q_list = pickle.load(open(filename, 'rb'))
    #A_list = pickle.load(open('./Sentences_As.pkl', 'rb'))

    end = time()
    print "Total loading time: %d seconds" % (end - begin)
    print "--------"

    nbatch = len(Q_list)/n
    if n % nbatch != 0:
        nbatch += 1

    for kk in range(n):

        start = kk * nbatch
        end = (kk + 1) * nbatch
        if end > len(Q_list):
            end = len(Q_list)

        temp_list = Q_list[int(start):int(end)]

        with open(filename[:-4]+'_%d.pkl'%(kk), 'wb') as f:
            pickle.dump(temp_list, f, -1)

    return


def Init_Sentences_from_list(word_list,dict):

   Qs=[]
   for line in word_list:


        sline0 = line.strip()
        rline0 = clean_face(sline0)
        rline0 = clean_notchinese(rline0)



        seg_list0 = jieba.cut(rline0)

        QQ = [w for w in seg_list0]


        temp_res_Q = [dict[w].index for w in QQ if w in dict]


        Qs.append(temp_res_Q)


   return Qs

if __name__ == '__main__':

    Split_Sentences('./Sentences_As.pkl')
    Split_Sentences('./Sentences_Qs.pkl')
'''
    # load word2vec model
    w2v_model = gensim.models.Word2Vec.load("./model/word2vec_gensim_100")

    word_vectors = w2v_model.wv

    dict=word_vectors.vocab
    dict_index2word=word_vectors.index2word

    sorted_vecs=[]
    for tmp_w in dict_index2word:
        tmp_vec=word_vectors[tmp_w]
        sorted_vecs.append(tmp_vec)

    sorted_vecs=np.asarray(sorted_vecs,dtype='float32')

    # load word2vec model
    data_path = './UsedData2'
    begin = time()
    Init_Sentences(data_path,dict)

    end = time()
    print "Total procesing time: %d seconds" % (end - begin)
'''

