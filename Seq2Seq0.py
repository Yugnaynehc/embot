# coding=utf-8
import sys
sys.path.append('.')

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import json
from time import time
from sklearn.externals import joblib
import gensim
import copy
import random

import numpy as np
import theano
import theano.tensor as T

from Make_Corporas import Init_Sentences_from_list


from lib import activations
from lib import updates
from lib import inits
from lib.theano_utils import floatX, sharedX
from lib.np_utils import np_softmax


# make result dir
desc = 'train_Seq2Seq_Bi_atten_0.3'  # train_Seq2Seq_LSTMhidden500  train_Seq2Seq

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
###################################

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


######################################
def encoder_network(Qs_words, Qs_masks, LSTM_hidden0, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc,
                    LSTM_hidden0_rev, W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev,
                    W_init_h0, b_init_h0, W_init_c0, b_init_c0,
                    U_attention_gen):

    # Qs_words: T * batch * word_dim
    # Qs_masks: T * batch

    LSTM_h0 = (T.extra_ops.repeat(LSTM_hidden0, repeats=Qs_words.shape[1], axis=0)).astype(
        theano.config.floatX)

    LSTM_h0_rev = (T.extra_ops.repeat(LSTM_hidden0_rev,
                   repeats=Qs_words.shape[1], axis=0)).astype(theano.config.floatX)

    cell0 = T.zeros((Qs_words.shape[1], n_LSTM), dtype=theano.config.floatX)

    ##################################################################

    # x_temp :  batch_size * dim_features
    def recurrence_enc(word_t, t_mask, h_t_prior, c_t_prior, W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc):

        lstm_t = T.dot(h_t_prior, W_LSTM_hidden_enc) + \
                       T.dot(word_t, W_LSTM_in_enc) + b_LSTM_enc
        i_t_enc = T.nnet.sigmoid(lstm_t[:, 0*n_LSTM:1*n_LSTM])
        f_t_enc = T.nnet.sigmoid(lstm_t[:, 1*n_LSTM:2*n_LSTM])

        cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2*n_LSTM:3*n_LSTM])
        cell_t_enc = t_mask.dimshuffle(
            [0, 'x']) * cell_t_enc + (1. - t_mask.dimshuffle([0, 'x'])) * c_t_prior

        o_t_enc = T.nnet.sigmoid(lstm_t[:, 3*n_LSTM:4*n_LSTM])
        h_t = o_t_enc * T.tanh(cell_t_enc)
        h_t = t_mask.dimshuffle([0, 'x']) * h_t + \
                                (1. - t_mask.dimshuffle([0, 'x'])) * h_t_prior


        return h_t.astype(theano.config.floatX), cell_t_enc.astype(theano.config.floatX)

    (h_list, _), _ = theano.scan(recurrence_enc, sequences=[Qs_words, Qs_masks],
                                                    outputs_info=[LSTM_h0, cell0],
                                                    non_sequences=[
                                                        W_LSTM_hidden_enc, W_LSTM_in_enc, b_LSTM_enc],
                                                    n_steps=Qs_words.shape[0],
                                                    strict=True)

    (h_list_rev, _), _ = theano.scan(recurrence_enc, sequences=[Qs_words[::-1, :, :], Qs_masks[::-1, :]],
                                                    outputs_info=[LSTM_h0_rev, cell0],
                                                    non_sequences=[
                                                        W_LSTM_hidden_enc_rev, W_LSTM_in_enc_rev, b_LSTM_enc_rev],
                                                    n_steps=Qs_words.shape[0],
                                                    strict=True)

    h_t_lang = T.concatenate([h_list, h_list_rev[::-1, :, :]], axis=2)  # was -1


    gen_init0_lang = T.concatenate([h_list[-1], h_list_rev[-1]], axis=1)

    # 1. for decoder
    LSTM_h0 = T.tanh(T.dot(gen_init0_lang, W_init_h0)+b_init_h0)
    cell0 = T.tanh(T.dot(gen_init0_lang, W_init_c0)+b_init_c0)

    word0 = (T.extra_ops.repeat(word_start, repeats=Qs_words.shape[1], axis=1)).astype(
        theano.config.floatX)

    hid_align = T.dot(h_t_lang, U_attention_gen)  # T_enc*Batch* dimAtten

    return h_t_lang, hid_align, LSTM_h0, cell0, word0  # T * batch * (2*n_LSTM)

######################################


def generate_next(h_t_prior, word_t_prior, c_t_prior, Qs_masks,  h_enc, hid_align,

                  W_LSTM_in_gen, W_LSTM_hidden_gen, b_LSTM_gen,
                  W_attention_gen, b_attention_gen, v_attention_gen,
                  W_word_gen, b_word_gen, W_softmax_gen, b_softmax_gen

                  ):  # x_temp :  batch_size * dim_features

        # calculate input

    lstm_t = T.dot(h_t_prior, W_LSTM_hidden_gen) + \
                   T.dot(word_t_prior, W_LSTM_in_gen) + b_LSTM_gen
    i_t_enc = T.nnet.sigmoid(lstm_t[:, 0*n_LSTM:1*n_LSTM])
    f_t_enc = T.nnet.sigmoid(lstm_t[:, 1*n_LSTM:2*n_LSTM])

    cell_t_enc = f_t_enc * c_t_prior + i_t_enc * T.tanh(lstm_t[:, 2*n_LSTM:3*n_LSTM])

    o_t_enc = T.nnet.sigmoid(lstm_t[:, 3*n_LSTM:4*n_LSTM])

    h_list = o_t_enc * T.tanh(cell_t_enc)

    h_t_info = T.concatenate([h_list, word_t_prior], axis=1)

    hdec_align = T.dot(h_t_info, W_attention_gen)  # *Batch* dimAtten

    all_align = T.tanh(
        hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_attention_gen.dimshuffle(['x', 'x', 0]))
    # T_enc  x batch_size x dimAttention

    e = all_align * v_attention_gen.dimshuffle(['x', 'x', 0])
    e = e.sum(axis=2) * Qs_masks  # T_enc  x batch_size
    # e = e.dimshuffle([1, 2, 0]) # T_dec x batch_size x T_enc

    # e2= T.reshape(e,[e.shape[0]*e.shape[1],e.shape[2]],ndim=2) # (T_dec x batch_size) x T_enc

    # normalize
    alpha = T.nnet.softmax(e.T)  # (batch_size) * T_enc

    # conv_feature representation at time T
    attention_enc = alpha.dimshuffle([1, 0, 'x']) * h_enc  # T_enc x batch_size x h_dim
    # T_dec x T_enc x batch_size x h_dim --> T_dec  x batch_size x h_dim
    attention_enc = attention_enc.sum(axis=0)

    prepare_word = T.concatenate([attention_enc, h_list], axis=1)

    word_t = lrelu(T.dot(prepare_word, W_word_gen) + b_word_gen)  # T * batch * middle_dim
    word_soft = T.dot(word_t, W_softmax_gen)+b_softmax_gen
    word_soft_K = T.nnet.softmax(word_soft)

    return word_soft_K.astype(theano.config.floatX), h_list.astype(theano.config.floatX), cell_t_enc.astype(theano.config.floatX)



####################################################
Qs_word_list = T.matrix('Qs_word_list', dtype='int32')  # batch * T
Qs_mask = T.matrix('Qs_mask', dtype='float32')  # batch * T

# provide Theano with a default test-value
Qs_word_list.tag.test_value = np.random.randint(
    1000, size=(nbatch, max_T)).astype(np.int32)

Qs_mask.tag.test_value = np.random.randint(1, size=(nbatch, max_T)).astype(np.int32)

####################################################
Qs_word_list_flat = T.flatten(Qs_word_list.T, ndim=1)
Qs_word_vecs = shared_Word_vecs[Qs_word_list_flat].reshape(
    [Qs_word_list.shape[1], Qs_word_list.shape[0], n_word_dim])  # T * batch * n_dim


h_t_lang, hid_align, LSTM_h0, cell0, word0 = encoder_network(
    Qs_word_vecs, Qs_mask, *enc_params)  # batch * n_LSTM

h_t_prior = T.matrix()
c_t_prior = T.matrix()
word_t_prior = T.matrix()

h_enc = T.tensor3()
hid_align_in = T.tensor3()

word_soft_K, h_t_next, c_t_next = generate_next(h_t_prior, word_t_prior, c_t_prior,
                                                Qs_mask, h_enc, hid_align_in,
                                                *gen_params)  # T *batch * n_word_dict


print 'COMPILING'
t = time()

_gen_init = theano.function([Qs_word_list, Qs_mask], [
                            h_t_lang, hid_align, LSTM_h0, cell0, word0])
_gen_next = theano.function([h_t_prior, word_t_prior, c_t_prior, Qs_mask, h_enc, hid_align_in], [
                            word_soft_K, h_t_next, c_t_next])

print '%.2f seconds to compile theano functions' % (time()-t)


######################################################
def generate_captions_perX(Qs_word_list, Qs_mask, Bsize):  # batch * T

    strategy = 3

    sample = []
    sample_score = []

    hyp_all_h_list, hyp_hid_align_list, hyp_h_list, hyp_c_list, hyp_word_list_embed = _gen_init(
        Qs_word_list, Qs_mask.T)  # h_enc, word0, cell0 ,  batch * X

    hyp_word_list_embed = hyp_word_list_embed.squeeze(axis=0)

    hyp_word_list = []

    hyp_scores = np.zeros((1,)).astype(theano.config.floatX)

    dead_k = 0
    live_k = 1

    for ii in range(max_T):

        # hyp_scores #B*Bsize

        word_soft_list, h_next_list, c_next_list = _gen_next(
            hyp_h_list, hyp_word_list_embed, hyp_c_list, Qs_mask.T, hyp_all_h_list, hyp_hid_align_list)

        voc_size = word_soft_list.shape[1]

        if strategy == 1:  # Max

            cand_scores = hyp_scores[:, None] - np.log(word_soft_list)
            cand_scores_flat = np.reshape(
                cand_scores, (cand_scores.shape[0]*cand_scores.shape[1],))

            ranks_flat = np.argsort(cand_scores_flat)[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy == 2:  # Softmax
            cand_scores = hyp_scores[:, None] + np.log(word_soft_list)
            cand_scores_flat = np.reshape(
                cand_scores, (cand_scores.shape[0]*cand_scores.shape[1],))
            cand_scores_flat = np_softmax(cand_scores_flat.astype('float64'))

            ranks_flat = np.random.multinomial(7*(Bsize-dead_k), cand_scores_flat, size=1)
            ranks_flat = np.argsort(-ranks_flat).squeeze()[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy == 3:  # 逐级加权

            jiaquan = 0.01
            cand_scores = hyp_scores[:, None]*jiaquan - np.log(word_soft_list)
            cand_scores_flat = np.reshape(
                cand_scores, (cand_scores.shape[0]*cand_scores.shape[1],))

            ranks_flat = np.argsort(cand_scores_flat)[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        elif strategy == 4:  # first_fix 首字符固定

            if ii == 1:
                hyp_scores = hyp_scores * 0

            cand_scores = hyp_scores[:, None] - np.log(word_soft_list)
            cand_scores_flat = np.reshape(
                cand_scores, (cand_scores.shape[0]*cand_scores.shape[1],))

            ranks_flat = np.argsort(cand_scores_flat)[:(Bsize-dead_k)]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size

        new_hyp_h_list = []
        new_hyp_c_list = []
        new_hyp_word_list_embed = []

        new_Qs_mask_list = []
        new_hyp_all_h_list_list = []
        new_hyp_hid_align_list_list = []

        new_hyp_scores = []
        new_hyp_word_list = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_h = copy.copy(h_next_list[ti, :])
            new_hyp_c = copy.copy(c_next_list[ti, :])
            new_Qs_mask = copy.copy(Qs_mask[ti, :])
            new_hyp_all_h_list = copy.copy(hyp_all_h_list[:, ti, :])
            new_hyp_hid_align_list = copy.copy(hyp_hid_align_list[:, ti, :])

            new_hyp_word_embed = sorted_vecs[wi]

            hyp_score = cand_scores[ti, wi]

            new_hyp_h_list.append(new_hyp_h)
            new_hyp_c_list.append(new_hyp_c)
            new_Qs_mask_list.append(new_Qs_mask)
            new_hyp_all_h_list_list.append(new_hyp_all_h_list)
            new_hyp_hid_align_list_list.append(new_hyp_hid_align_list)

            new_hyp_word_list_embed.append(new_hyp_word_embed)
            new_hyp_scores.append(hyp_score)

            if len(hyp_word_list) == 0:
                temp_hyp_word_list = [wi]
            else:
                temp_hyp_word_list = copy.copy(hyp_word_list[ti])
                temp_hyp_word_list.append(wi)

            new_hyp_word_list.append(temp_hyp_word_list)

        new_live_k = 0

        hyp_h_list = []
        hyp_c_list = []
        # hyp_alpha_list=[]
        Qs_mask = []
        hyp_all_h_list = []
        hyp_hid_align_list = []

        hyp_word_list_embed = []

        hyp_scores = []
        hyp_word_list = []

        for idx in range(len(new_hyp_word_list)):
            if new_hyp_word_list[idx][-1] == voc_size-1:
                sample.append(new_hyp_word_list[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k = new_live_k+1
                hyp_h_list.append(new_hyp_h_list[idx])
                hyp_c_list.append(new_hyp_c_list[idx])
                # hyp_alpha_list.append(new_hyp_alpha_list[idx])
                Qs_mask.append(new_Qs_mask_list[idx])
                hyp_all_h_list.append(new_hyp_all_h_list_list[idx])
                hyp_hid_align_list.append(new_hyp_hid_align_list_list[idx])

                hyp_word_list_embed.append(new_hyp_word_list_embed[idx])

                hyp_scores.append(new_hyp_scores[idx])
                hyp_word_list.append(new_hyp_word_list[idx])

        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= Bsize:
            break

        hyp_scores = np.array(hyp_scores).astype(theano.config.floatX)

        hyp_h_list = np.array(hyp_h_list).astype(theano.config.floatX)
        hyp_c_list = np.array(hyp_c_list).astype(theano.config.floatX)
       # hyp_alpha_list=np.array(hyp_alpha_list).astype(theano.config.floatX)
        Qs_mask = np.array(Qs_mask).astype(theano.config.floatX)
        hyp_all_h_list = np.array(hyp_all_h_list).astype(theano.config.floatX)
        hyp_hid_align_list = np.array(hyp_hid_align_list).astype(theano.config.floatX)

        if hyp_all_h_list.ndim == 3:
            hyp_all_h_list = np.transpose(hyp_all_h_list, (1, 0, 2))
            hyp_hid_align_list = np.transpose(hyp_hid_align_list, (1, 0, 2))
        else:
            hyp_all_h_list = np.expand_dims(hyp_all_h_list, axis=1)
            hyp_hid_align_list = np.expand_dims(hyp_hid_align_list, axis=1)

        hyp_word_list_embed = np.array(hyp_word_list_embed).astype(theano.config.floatX)
        if hyp_word_list_embed.ndim == 1:
            hyp_word_list_embed = np.expand_dims(hyp_word_list_embed, axis=0)

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_word_list[idx])
            sample_score.append(hyp_scores[idx])

    return sample, sample_score

######################################################


def prepare_files_Q(Qs_batch, word_end_inx):

    word_end_inx = word_end_inx-1

    Qs_lens = [len(tl) for tl in Qs_batch]

    max_Qs = max(Qs_lens)

    batch_Q_word_list = []
    batch_Q_word_list_reverse = []
    batch_Q_mask_list = []

    for tll in range(len(Qs_batch)):

        temp_s = Qs_batch[tll]
        temp_len = len(temp_s)
        word_list = np.concatenate(
            (np.asarray(temp_s, dtype='int32'), word_end_inx*np.ones(max_Qs-temp_len, dtype='int32')))
        word_list_reverse = np.concatenate((np.asarray(temp_s, dtype='int32')[
                                           ::-1], word_end_inx*np.ones(max_Qs-temp_len, dtype='int32')))
        mask_list = np.concatenate(
            (np.ones(temp_len, dtype='int32'), np.zeros(max_Qs-temp_len, dtype='int32')))

        batch_Q_word_list.append(word_list)
        batch_Q_word_list_reverse.append(word_list_reverse)
        batch_Q_mask_list.append(mask_list)

    return np.asarray(batch_Q_word_list, dtype='int32'), np.asarray(batch_Q_word_list_reverse, dtype='int32'), np.asarray(batch_Q_mask_list, dtype='float32')


def chat(ques):
    test_words_list = [ques]

    Q_list = Init_Sentences_from_list(test_words_list,dict)

    Beam=30

    n = len(Q_list)
    batches = n / nbatch
    if n % nbatch != 0:
        batches += 1

    total_captions=[]

    for kk in range(batches):

        start = kk * nbatch
        end = (kk + 1) * nbatch
        if end > n:
            end = n

        Q_in = [Q_list[i] for i in range(start,end)]

        W1, W2, W3 = prepare_files_Q(Q_in,n_word_dict)

        test_gen_captions, scores = generate_captions_perX(W1,W3, Beam)
        print scores

        # avg_scores = [len(test_gen_captions[idx]) / scores[idx] for idx in range(len(scores))]
        # ranks = np.argsort(avg_scores)
        # ranks = np.argsort(scores)
        # print ranks
        # test_gen_captions = [test_gen_captions[idx] for idx in ranks]

        temp_dict = {}
        temp_dict['Q']=test_words_list[kk]

        temp_list = []
        for ii in range(len(test_gen_captions)):

            temp_words = test_gen_captions[ii]
            temp_str = ''
            for jj in range(len(temp_words)):
                temp_word = dict_index2word[temp_words[jj].astype(np.int32)]
                if temp_word != u'EOF':
                    temp_str = temp_str + temp_word.encode('utf8')  #.encode('utf-8')
                else:
                    break
            temp_list.append(temp_str)

        # print temp_list
        temp_dict['QAs']=temp_list

        total_captions.append(temp_dict)

    # end = time()
    # print "test time: %d seconds" % (end - begin)
    # print "--------"

    # file = './test_res_%s_epoach%d_group%d.txt'%(desc,select_epochs,select_groups)

    # fp = open(file,'w')
    # fp.write(json.dumps(total_captions, ensure_ascii=False))
    # fp.close()

    res = json.dumps(total_captions, ensure_ascii=False)
    # temp_list = temp_list[5:-5]
    print res
    # temp_list = sorted(temp_list, key=lambda x: len(x))
    # temp_list = temp_list[-3:]
    # random.shuffle(temp_list)
    # return temp_list[0]
    filter_list = [x for x in temp_list if 12 < len(x) < 45]
    random.shuffle(filter_list)
    print res
    if len(filter_list) > 0:
        return filter_list[-1]
    else:
        return temp_list[-1]
