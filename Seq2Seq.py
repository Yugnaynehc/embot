# coding=utf-8
import gensim
import logging
import multiprocessing
import os
import re
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
# sys.setdefaultencoding('utf-8')
# sys.setdefaultencoding('gbk')
import jieba
import numpy as np
import copy

import codecs
import six.moves.cPickle as pickle
from gensim.models.keyedvectors import KeyedVectors
from gensim import corpora, models, similarities
from time import time
import random
from collections import OrderedDict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# Used for As_Q
from sklearn.externals import joblib
#import six.moves.cPickle as pickle
import gensim
import copy

import theano
import theano.tensor as T
from Make_Corporas import Init_Sentences_from_list


from lib import activations
from lib import updates
from lib import inits
#from lib.vis import color_grid_vis
#from lib.rng import py_rng, np_rng,t_rng,t_rng_cpu
#from lib.ops import batchnorm
from lib.theano_utils import floatX, sharedX
from lib.np_utils import np_softmax
# Used for As_Q


# load Word2Vec model
print os.curdir
model = gensim.models.Word2Vec.load(os.path.join("model", "word2vec_gensim"))

word_vectors = model.wv

dict = word_vectors.vocab
dict_index2word = word_vectors.index2word

sorted_vecs = []
for tmp_w in dict_index2word:
    tmp_vec = word_vectors[tmp_w]
    sorted_vecs.append(tmp_vec)

sorted_vecs = np.asarray(sorted_vecs, dtype='float32')

dict_index2word.append(u'EOF')
sorted_vecs = np.concatenate(
    (sorted_vecs, 7 * np.ones((1, sorted_vecs.shape[1]), dtype='float32')), axis=0)


n_word_dict = sorted_vecs.shape[0]
n_word_dim = sorted_vecs.shape[1]  # # of dim of word representation
####################################################
# params
nbatch = 1      # # of examples in batch
max_T = 10          # # sentense length
n_LSTM = 300       # # of LSTM_hidden_units    #1000 1500
dimAttention = 100

# init settings
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()


orfn = inits.Orthogonal(scale=1)
gifn = inits.Normal(scale=0.01)

gain_ifn = inits.Normal(loc=1., scale=0.01)
bias_ifn = inits.Constant(c=0.)

startword_ifn = inits.Constant(c=-7.)
# load As_Q test model

# make result dir
desc2 = 'train_Seq2Seq_p(xy)'  # train_Seq2Seq_LSTMhidden500  train_Seq2Seq
select_epochs = 8

#word_start = startword_ifn((1, 1, n_word_dim), 'word_start')
shared_Word_vecs = sharedX(sorted_vecs)  # T._shared(sorted_vecs, borrow=True)

[LSTM_hidden02, W_LSTM_hidden_enc2, W_LSTM_in_enc2, b_LSTM_enc2,
 W_LSTM_hidden_gen2, W_LSTM_in_gen2, b_LSTM_gen2, W_word_gen2, b_word_gen2, W_softmax_gen2, b_softmax_gen2] = \
    [sharedX(p) for p in joblib.load('models/%s/%d_total_params.jl' % (desc2, select_epochs))]
#[sharedX(p) for p in joblib.load('models/%s/%d_%d_%d_total_params.jl' % (desc, select_epochs, select_groups, select_steps))]

enc_params2 = [LSTM_hidden02, W_LSTM_hidden_enc2, W_LSTM_in_enc2, b_LSTM_enc2]

gen_params2 = [W_LSTM_hidden_gen2, W_LSTM_in_gen2, b_LSTM_gen2,
               W_word_gen2, b_word_gen2, W_softmax_gen2, b_softmax_gen2]


'''
select_epochs = 4
select_groups = 0

word_start = startword_ifn((1, 1, n_word_dim), 'word_start')

shared_Word_vecs = sharedX(sorted_vecs)  # T._shared(sorted_vecs, borrow=True)

[LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
 LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev, U_attention_gen, W_attention_gen, b_attention_gen, v_attention_gen,
 W_init_h0, b_init_h0, W_init_c0, b_init_c0,
 W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen, W_word_gen, b_word_gen, W_softmax_gen, b_softmax_gen] = \
    [sharedX(p) for p in joblib.load('models/%s/%d_%d_total_params.jl' %
             (desc, select_epochs, select_groups))]

enc_params = [LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
                LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev,
                W_init_h0, b_init_h0, W_init_c0, b_init_c0,
                U_attention_gen]

gen_params = [W_LSTM_in_gen, W_LSTM_hidden_gen, b_LSTM_gen,
              W_attention_gen, b_attention_gen, v_attention_gen,
              W_word_gen, b_word_gen, W_softmax_gen, b_softmax_gen]

total_params = []
total_params.extend(enc_params)
total_params.extend(gen_params)
'''

# define functions 2


def encoder_network2(Qs_words, Qs_masks, LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc):

    # Qs_words: T * batch * word_dim
    # Qs_masks: T * batch

    # Word_list_reverse=Word_list[::-1,:,:]
    # hid_align=T.dot(Xs,U_attention_dis)
    #hid_align = Xs.dimshuffle([0,1,2,'x']) * U_attention_dis.dimshuffle(['x','x',0,1])
    # hid_align = hid_align.sum(axis=2) # conv_feature_length x batch_size x dimAttention # was -2
    # X_mean=Xs.mean(axis=0) #  (W*H)*batch*C
    #X_mean0=relu(T.dot(X_mean, W_init_mean0)+b_init_mean0)

    LSTM_h0 = (T.extra_ops.repeat(LSTM_hidden0, repeats=Qs_words.shape[1], axis=0)).astype(theano.config.floatX)
    cell0 = T.zeros((Qs_words.shape[1], n_LSTM), dtype=theano.config.floatX)

    ##################################################################

    # x_temp :  batch_size * dim_features
    def recurrence_enc(word_t, t_mask, h_t_prior, c_t_prior, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc):

        # h_t_atten = T.concatenate([h_t_prior, word_t,alpha_prior], axis=1) # was -1
        # hdec_align = T.dot(h_t_atten, W_attention_dis) # batch_size x dimAttention
        # all_align = T.tanh(hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_attention_dis.dimshuffle(['x','x',0])) # conv_feature_length x batch_size x dimAttention

        #e = all_align * v_attention_dis.dimshuffle(['x','x',0])
        # e = e.sum(axis=2) # conv_feature_length x batch_size # was -1

        # normalize
        # alpha = T.nnet.softmax(e.T) # conv_feature_length x batch_size

        # conv_feature representation at time T
        # attention_img = alpha.dimshuffle([1, 0, 'x']) * Xs #conv_feature_length x batch_size x img_conv_dims
        # attention_img = attention_img.sum(axis=0) # batch_size x img_conv_dims

        ###########################################################################
        #LSTM_in_t=T.concatenate([attention_img,word_t], axis=1)

        lstm_t = T.dot(h_t_prior, W_LSTM_hidden_enc) + T.dot(word_t, W_LSTM_in_enc) + b_LSTM_enc
        i_t_enc = T.nnet.sigmoid(lstm_t[:, 0 * n_LSTM:1 * n_LSTM])
        f_t_enc = T.nnet.sigmoid(lstm_t[:, 1 * n_LSTM:2 * n_LSTM])

        cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2 * n_LSTM:3 * n_LSTM])
        cell_t_enc = t_mask.dimshuffle([0, 'x']) * cell_t_enc + (1. - t_mask.dimshuffle([0, 'x'])) * c_t_prior

        o_t_enc = T.nnet.sigmoid(lstm_t[:, 3 * n_LSTM:4 * n_LSTM])
        h_t = o_t_enc * T.tanh(cell_t_enc)
        h_t = t_mask.dimshuffle([0, 'x']) * h_t + (1. - t_mask.dimshuffle([0, 'x'])) * h_t_prior

        #y_t=sigmoid(T.dot(h_t, W_dis) + b_dis)

        return h_t.astype(theano.config.floatX), cell_t_enc.astype(theano.config.floatX)

    (h_list, cell_list), updates_LSTM = theano.scan(recurrence_enc,
                                                    sequences=[Qs_words, Qs_masks],
                                                    outputs_info=[LSTM_h0, cell0],
                                                    non_sequences=[W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc],
                                                    n_steps=Qs_words.shape[0],
                                                    strict=True)

    return h_list[-1]

######################################


def generate_captions2(As_words, As_masks, h_enc, W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen, W_word_gen, b_word_gen, W_softmax_gen, b_softmax_gen):

    # As_words: T * batch * word_dim
    # As_masks: T * batch

    # hid_align=T.dot(Xs,U_attention_gen)

    # X_mean=Xs.mean(axis=0)
    #X_mean0=relu(T.dot(X_mean, W_init_mean0)+b_init_mean0)
    # RELU dropout
    # X_mean0=dropout_layer(X_mean0,use_noise,t_rng)

    #LSTM_h0=T.tanh(T.dot(X_mean0, W_init_h0)+b_init_h0)
    #cell0=T.tanh(T.dot(X_mean0, W_init_c0)+b_init_c0)
    # alpha_0=(1.0/(n_regins*n_regins))*T.ones((Xs.shape[1],n_regins*n_regins),dtype=theano.config.floatX)

    word0 = -7 * T.ones((1, As_words.shape[1], n_word_dim), dtype=theano.config.floatX)
    this_real_words = T.concatenate([word0, As_words], axis=0)

    cell0 = T.zeros((As_words.shape[1], n_LSTM), dtype=theano.config.floatX)

    def recurrence(word_t_prior, t_mask, h_t_prior, c_t_prior, W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen):  # x_temp :  batch_size * dim_features

        # calculate input

        lstm_t = T.dot(h_t_prior, W_LSTM_hidden_gen) + T.dot(word_t_prior, W_LSTM_in_gen) + b_LSTM_gen
        i_t_enc = T.nnet.sigmoid(lstm_t[:, 0 * n_LSTM:1 * n_LSTM])
        f_t_enc = T.nnet.sigmoid(lstm_t[:, 1 * n_LSTM:2 * n_LSTM])

        cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2 * n_LSTM:3 * n_LSTM])
        cell_t_enc = t_mask.dimshuffle([0, 'x']) * cell_t_enc + (1. - t_mask.dimshuffle([0, 'x'])) * c_t_prior

        o_t_enc = T.nnet.sigmoid(lstm_t[:, 3 * n_LSTM:4 * n_LSTM])

        h_t = o_t_enc * T.tanh(cell_t_enc)
        h_t = t_mask.dimshuffle([0, 'x']) * h_t + (1. - t_mask.dimshuffle([0, 'x'])) * h_t_prior

        return h_t.astype(theano.config.floatX), cell_t_enc.astype(theano.config.floatX)

    (h_list, cell_list), updates_LSTM2 = theano.scan(recurrence,
                                                     sequences=[this_real_words[0:-1], As_masks],
                                                     outputs_info=[h_enc, cell0],
                                                     non_sequences=[W_LSTM_hidden_gen, W_LSTM_in_gen, b_LSTM_gen],
                                                     n_steps=As_masks.shape[0],
                                                     strict=True)

    #prepare_word=T.concatenate([h_list,this_real_words[0:-1],atten_img_list], axis=2)
    # dropout
    word_t = lrelu(T.dot(h_list, W_word_gen) + b_word_gen)  # T * batch * middle_dim
    # dropout
    word_soft = T.dot(word_t, W_softmax_gen) + b_softmax_gen
    word_soft_K = T.nnet.softmax(
        T.reshape(word_soft, [word_soft.shape[0] * word_soft.shape[1], word_soft.shape[2]], ndim=2))

    return word_soft_K  # (T *batch ) * n_word_dict


######################################################
####################################################
Qs_word_list2 = T.matrix('Qs_word_list2', dtype='int32')  # batch * T
Qs_mask2 = T.matrix('Qs_mask2', dtype='float32')  # batch * T
As_word_list2 = T.matrix('As_word_list2', dtype='int32')  # batch * T
As_mask2 = T.matrix('As_mask2', dtype='float32')  # batch * T

# provide Theano with a default test-value
#Qs_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,max_T)).astype(np.int32)
#As_word_list.tag.test_value = np.random.randint(1000,size=(nbatch,max_T)).astype(np.int32)

#Qs_mask.tag.test_value = np.random.randint(1,size=(nbatch,max_T)).astype(np.int32)
#As_mask.tag.test_value = np.random.randint(1,size=(nbatch,max_T)).astype(np.int32)

####################################################
Qs_word_list_flat2 = T.flatten(Qs_word_list2.T, ndim=1)
Qs_word_vecs2 = shared_Word_vecs[Qs_word_list_flat2].reshape(
    [Qs_word_list2.shape[1], Qs_word_list2.shape[0], n_word_dim])  # T * batch * n_dim

As_word_list_flat2 = T.flatten(As_word_list2.T, ndim=1)  # words x #samples
As_word_vecs2 = shared_Word_vecs[As_word_list_flat2].reshape(
    [As_word_list2.shape[1], As_word_list2.shape[0], n_word_dim])  # T * batch * n_dim


h_enc2 = encoder_network2(Qs_word_vecs2, Qs_mask2.T, *enc_params2)  # batch * n_LSTM

word_K_list2 = generate_captions2(As_word_vecs2, As_mask2.T, h_enc2, *gen_params2)  # T *batch * n_word_dict

# word_K_list: (T *batch ) * n_word_dict

word_K_list_flat2 = T.flatten(word_K_list2, ndim=1)
# tensor.arange(x_flat.shape[0])   *  probs.shape[1]  +  x_flat
cost = T.log(word_K_list_flat2[T.arange(As_word_list_flat2.shape[0]) * n_word_dict + As_word_list_flat2] + 1e-7)
cost_re = T.reshape(cost, [As_word_list2.shape[1], As_word_list2.shape[0]], ndim=2)  # T *batch

cost1 = cost_re * As_mask2.T  # T *batch
cost2 = cost1.sum(axis=0)  # /Mask_captions.sum(axis=0)
# cost3=cost2.mean()


#lrt = sharedX(learning_rate)
#g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2),clipnorm=10)

#g_updates = g_updater(total_params, cost3)


print 'COMPILING'
t = time()

_train_score = theano.function([Qs_word_list2, As_word_list2, Qs_mask2, As_mask2], [cost2])
# mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
#_train_d = theano.function([X_img_conv,Zs,Real_captions,Mask_captions,Fake_img_conv], [d_cost,d_cost_real,d_cost_gen,d_cost_fake,gen_captions],
# updates=d_updates)
# mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
print '%.2f seconds to compile theano functions' % (time() - t)


######################################################
#####################################
def prepare_files(Qs_batch, As_batch, word_end_inx):

    word_end_inx = word_end_inx - 1

    Qs_lens = [len(tl) for tl in Qs_batch]
    As_lens = [len(tl) for tl in As_batch]

    max_Qs = max(Qs_lens)
    max_As = max(As_lens) + 1

    batch_Q_word_list = []
    batch_Q_word_list_reverse = []
    batch_Q_mask_list = []

    batch_A_word_list = []
    batch_A_mask_list = []

    for tll in range(len(Qs_batch)):

        temp_s = Qs_batch[tll]
        temp_len = len(temp_s)
        word_list = np.concatenate((np.asarray(temp_s, dtype='int32'), word_end_inx *
                                    np.ones(max_Qs - temp_len, dtype='int32')))
        word_list_reverse = np.concatenate(
            (np.asarray(temp_s, dtype='int32')[::-1], word_end_inx * np.ones(max_Qs - temp_len, dtype='int32')))
        mask_list = np.concatenate((np.ones(temp_len, dtype='int32'), np.zeros(max_Qs - temp_len, dtype='int32')))

        batch_Q_word_list.append(word_list)
        batch_Q_word_list_reverse.append(word_list_reverse)
        batch_Q_mask_list.append(mask_list)

        temp_s = As_batch[tll]
        temp_len = len(temp_s)
        word_list = np.concatenate((np.asarray(temp_s, dtype='int32'), word_end_inx *
                                    np.ones(max_As - temp_len, dtype='int32')))

        mask_list = np.concatenate((np.ones(temp_len + 1, dtype='int32'),
                                    np.zeros(max_As - temp_len - 1, dtype='int32')))

        batch_A_word_list.append(word_list)
        batch_A_mask_list.append(mask_list)

    return np.asarray(batch_Q_word_list, dtype='int32'), np.asarray(batch_Q_word_list_reverse, dtype='int32'), np.asarray(batch_Q_mask_list, dtype='float32'), np.asarray(batch_A_word_list, dtype='int32'), np.asarray(batch_A_mask_list, dtype='float32')
# Used for As_Q


def clean_aa(_str):
    _list = list(_str.decode('utf8'))
    n = len(_list)
    list1 = []
    if n <= 1:
        return _str.encode('utf8')
    for i in range(n - 1):
        if _list[i] != _list[i + 1]:
            list1.append(_list[i].encode('utf8'))
    list1.append(_list[-1].encode('utf8'))
    str1 = ''.join(list1)
    return str1


def clean_abab_set(_str):
    seg_list = jieba.cut(_str)
    _list = [w for w in seg_list]
    #str1 = ''.join(list(OrderedDict.fromkeys(_list)))
    news_ids = list(set(_list))
    news_ids.sort(key=_list.index)
    str1 = ''.join(news_ids)
    return str1


#########################################################################################init and loading
print "Loading dict --------"
filename = './models/LDA2Vec_dict.dict'
dictionary = corpora.Dictionary.load(filename)
print "--------"

print "Loading Tfidf model --------"
model_fname = './models/Tfidf2Vec_trained_model_iter1.l2v'
Tfidf_model = gensim.models.TfidfModel.load(model_fname)
print "--------"

print "Loading Similarity index model --------"
index = similarities.SparseMatrixSimilarity.load('./models/Tfidf2Vec_index.index')

'''
print "Loading data --------"
filename='./models/LDA2Vec_corpus.pkl'
train_corpus = pickle.load(open(filename, 'rb'))
print "--------"
'''
print "Loading answers data --------"
filename = './models/LDA2Vec_answers.pkl'
train_answers = pickle.load(open(filename, 'rb'))
print "--------"

# print "Loading qs data --------"
# filename = './models/LDA2Vec_qs.pkl'
# train_qs = pickle.load(open(filename, 'rb'))
# print "--------"
###########################################################################################


def chat(ask):
    ###########################Step 1: search IFIDF results ###########################

    index.num_best = 300

    seg_list = jieba.cut(ask)
    temp_res = [w for w in seg_list]

    vec_bow = dictionary.doc2bow(temp_res)
    vec_lda = Tfidf_model[vec_bow]  # convert the query to LDA space

    begin = time()
    sims = index[vec_lda]  # perform a similarity query against the corpus
    end1 = time()

    sims = sorted(sims, key=lambda item: -item[1])
    end2 = time()

    total_As_list = []
    for ii in range(len(sims)):

        '''
        temp_str=''
        for ss in train_qs[sims[ii][0]]:   ############################################# 选择匹配的对象  all   Q   As
            temp_str=temp_str+ss.encode('utf8')
        total_As_list.append(temp_str)

        temp_str=''
        for ss in train_corpus[sims[ii][0]]:   ############################################# 选择匹配的对象  all   Q   As
            temp_str=temp_str+ss.encode('utf8')
        total_As_list.append(temp_str)
        '''
        temp_str = ''
        for ss in train_answers[sims[ii][0]]:  # 选择匹配的对象  all   Q   As
            temp_str = temp_str + ss.encode('utf8')
        total_As_list.append(temp_str)
        #print(u'%s: «%s»\n' % (sims[ii], ' '.join(train_corpus[sims[ii][0]])))

    ###########################Step 2: sort by  IFIDF lengths  ###########################

    select_rate2 = 0.4

    str_order_dict = OrderedDict()
    for temp_str in total_As_list:
        cleaned_str = clean_aa(clean_abab_set(copy.copy(temp_str)))
        str_order_dict[cleaned_str] = temp_str

    new_list = str_order_dict.keys()
    sum_list = []
    for ii in new_list:
        seg_list = jieba.cut(ii)
        temp_res = [w for w in seg_list]

        vec_bow = dictionary.doc2bow(temp_res)
        vec_Tfidf = Tfidf_model[vec_bow]  # convert the query to LDA space
        sum_list.append(sum([x[1] for x in vec_Tfidf]))
        #print(u'%s \n' % (ii))
    ranks = np.argsort(np.array(sum_list) * -1)

    #print(u'SIMILAR/DISSIMILAR DOCS: %s:\n' % ask)

    result_list2 = []
    for ii in range(int(len(ranks) * select_rate2)):
        #print(u'%s \n' % (str_order_dict[new_list[ranks[ii]]]))
        result_list2.append(str_order_dict[new_list[ranks[ii]]])

    ###########################Step 3: sort by  As_Q scores  ###########################
    select_rate3 = 0.1

    # p(X|Y) used for re-ranking
    Q_list = Init_Sentences_from_list([ask], dict)
    A_list = Init_Sentences_from_list(result_list2, dict)

    Q_in = [Q_list[0] for i in range(len(A_list))]
    A_in = A_list

    #W1, W2, W3, W4, W5 = prepare_files(Q_in, A_in,n_word_dict)
    W1, W2, W3, W4, W5 = prepare_files(A_in, Q_in, n_word_dict)  # train p(X|Y) used for re-ranking   date:  2017.8.25

    MSE_score = _train_score(W1, W4, W3, W5)
    MSE_score = np.squeeze(MSE_score) * -1
    ranks = np.argsort(MSE_score)
    ranks = list(np.squeeze(ranks))

    #sims = sorted(sims, key=lambda item: -item[1])
    #end2 = time()
    # print "Total Testing time: %4f seconds" % (end1 - begin)
    # print "Total Sorting time: %4f seconds" % (end2 - begin)

    #results=[train_corpus[sims[ii][0]].words for ii in range(20)]
    print(u'SIMILAR/DISSIMILAR DOCS: %s:\n' % ask)
    show_str_list = []
    for ii in range(int(len(ranks) * select_rate3)):
        print(u'%s: %d  %s\n' % (MSE_score[int(ranks[ii])], int(ranks[ii]), result_list2[int(ranks[ii])]))
        #print(u'%s: %d  %s\n' % (MSE_score[int(ranks[ii])],int(ranks[ii]) ,clean_aa(clean_abab_set(total_As_list[int(ranks[ii])]))))
        show_str_list.append(result_list2[int(ranks[ii])])

    random.shuffle(show_str_list)

    return show_str_list[-1]
