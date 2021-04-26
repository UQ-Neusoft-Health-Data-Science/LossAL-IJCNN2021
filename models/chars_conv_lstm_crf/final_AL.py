"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"
import torch.nn as nn
import functools
import json
import logging
from pathlib import Path
import sys
from scipy import sparse
import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
import os
import utils.file_util as file_util
import utils.data_util as data_util
import utils.semtype_util as semtype_util
import utils.text_util as text_util
import utils.regression_util as regression_util
import utils.crf as crf
import shutil
import datetime
from datetime import datetime
import math
from masked_conv import masked_conv1d_and_max
import utils.gcn_util as gcn_util
tf.enable_eager_execution()
import random
import time
import re
from itertools import chain
from tensorflow.python.ops import array_ops
import collections
import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
data_name='clef'
# output_dir="/home/s4616573/data/bert_ner_output/"+data_name+"_loss_delta_0/"
'''SEED SET NUMBER'''
seed_set_num=4
ite_num=65
DATADIR = '../../data/example'
DATADIR = 'data/example'
simulation_folder = "/home/s4616573/data/bert_ner_output/data_simulation_al_gcn/"
undefined_value=pow(math.e,-10)
uncertain_threshold=0.1
alpha_const=0.75
beta_const=0.5
output_dir="/home/s4616573/data/bert_ner_output/"+data_name+"_alphabeta2/"
# use_positive_loss=False
regression_output_dir=output_dir+"regression/"
use_LC=False
use_MNLP=True
use_LCC=False

# Logging
use_sentence_base=True
import torch.nn.functional as F
# logging.basicConfig(filename=output_dir+"logging_al.txt",filemode='a',level=logging.INFO)
initial_loss_value=9999

use_i2b2=True if data_name=='i2b2' else False
use_clef=True if data_name=='clef' else False
use_conll=True if data_name=='conll' else False
if use_i2b2:
    dataset_dir="/home/s4616573/data/i2b2/"
if use_clef:
    dataset_dir="/home/s4616573/data/CLEF/"

if use_conll:
    dataset_dir = "/home/s4616573/data/conll-corpora/conll2003/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    shutil.rmtree(output_dir,ignore_errors=True)
    os.mkdir(output_dir)
Path(output_dir+'results').mkdir(exist_ok=True)
tf.compat.v1.logging.set_verbosity(logging.INFO)
log_file=output_dir+'results/'+data_name+'_alpha'+str(alpha_const)+'_beta'+str(beta_const)+'_seed'+str(seed_set_num)+'.txt'
handlers = [
    logging.FileHandler(log_file),
    logging.StreamHandler(sys.stdout)
]
# tf.nn.sigmoid_cross_entropy_with_logits
logging.getLogger('tensorflow').handlers = handlers
def get_fscores():
    data = open(log_file, "r").readlines()
    # data=file_util.load(file)
    # f_scores = [float(x.split('=')[-1].strip()) if 'F1:' not in x else float(x.split(':')[-1].strip()) for x in data if 'eval_f' in x or 'F1:' in x]# or 'fx:' in x#BERT
    f_scores = [float(x.split(',')[1].split('=')[-1].strip()) if 'F1:' not in x else float(x.split(':')[-1].strip()) for
                x in data if
                'f1 =' in x or 'F1:' in x]  # or 'fx:'
    return  f_scores
def train_regression_loss(params):
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120, tf_random_seed=12)
    if os.path.exists(regression_output_dir):
        shutil.rmtree(regression_output_dir, ignore_errors=True)
        os.mkdir(regression_output_dir)
    reg_estimator = tf.estimator.Estimator(regression_model_fn,regression_output_dir, cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    # hook = tf.estimator.experimental.stop_if_no_increase_hook(
    #     reg_estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    # print("trained_concept_count", params["trained_concept_count"])
    # do_train_eval(estimator,selected_texts, selected_tags)
    do_regression_train_eval(reg_estimator, params["labelled_texts"], params["rank_labels"])
    unlabelled_texts = params["unlabelled_texts"]
    params["predict_mode"]=True
    loss_fake_predict = [0.00] * len(unlabelled_texts)
    loss_fake_predict=np.reshape(loss_fake_predict, (-1,1))
    # params["unlabelled_texts"]=params["labelled_texts"]

    # test_inpf = functools.partial(selected_regression_input_fn,unlabelled_texts ,loss_fake_predict,params)
    test_inpf = functools.partial(selected_regression_input_fn, unlabelled_texts ,loss_fake_predict , params)
    # results = reg_estimator.predict(test_inpf,yield_single_examples=True)#, yield_single_examples=True

    # test_inpf = functools.partial(input_fn, fwords('clef_train'), ftags('clef_train'))
    results = reg_estimator.predict(test_inpf,yield_single_examples=True)
    unlabelled_loss_list=[]
    uresult_counter=0
    current_processed=(0,0)
    start_idx=0
    loss_bar=[]
    loss_bar_list=[]
    # unlabelled_tokens=text_util.count_token(params["all_text_data"])
    ucounter=0
    unlabelled_idx=params["unlabelled_idx"]
    delta_dict=params["delta_dict"]
    for result in results:
        text_x = unlabelled_texts[ucounter].split()
        loss_x = result["pred_ids"][:len(text_x)]
        loss_i = np.array(result["pred_ids"][:len(text_x)])
        loss_i = np.reshape(loss_i,(len(text_x)))
        last_loss = delta_dict[unlabelled_idx[ucounter]][0]
        last_delta = delta_dict[unlabelled_idx[ucounter]][1]
        delta_loss_i = alpha_const * loss_i + (1 - alpha_const) * (loss_i - last_loss)
        current_delta_loss = loss_i - last_loss
        current_delta_delta = current_delta_loss - last_delta

        current_delta_loss = [1 if x > 0 else 0 for x in delta_loss_i]
        # delta_delta_final = np.multiply(current_delta_loss, current_delta_delta)
        # delta_delta_final = [x if x > 0 else 0 for x in delta_delta_final]
        # delta_delta_final =0.7*np.array([x if x>0 else 0 for x in loss_i-last_loss]) +0.3*np.array([x if x > 0 else 0 for x in delta_delta_final])
        delta_delta_final = np.array([x  for x in loss_i - last_loss])#if x > 0 else 0

        # if use_positive_loss:
        #     delta_loss_i=[x if x >0 else 0 for x in delta_loss_i]
        # else:
        #     delta_loss_i = [x if x <0 else 0 for x in delta_loss_i]

        delta_dict[unlabelled_idx[ucounter]] = [loss_i, loss_i - last_loss]
        # loss_list.append(delta_delta_final)


        # loss_bar.extend(loss_x)
        # loss_x=[x  for t,x in enumerate(loss_x) if x >0]

            # sample_loss=abs(np.amin(loss_x)) if len([x for x in loss_x if x <0])>0 else np.amax(loss_x)#/len([x for x in loss_x if x>0])
            # sample_loss=np.sum([x for x in delta_delta_final if x>0]) if len([x for x in delta_delta_final if x>0])>0 else 0#/len(text_x)
        if alpha_const==1:
                if len(loss_i) > 0:
                    sample_loss = np.sum([x for x in loss_i if x!=0] )#/len(text_x)
                    uncertain_len = len([x for x in loss_i if x != 0])
                else:
                    sample_loss=0
                    uncertain_len=1
            # if sample_loss>0:
            #     print("ha ha ha")
        else:
            if len([x for x in delta_loss_i if x>0 ])>0:
                    sample_loss = np.sum([x for x in delta_loss_i if x>0 ] ) #/len(text_x)#/len(text_x)
                    uncertain_len =len([x for x in delta_loss_i if x>0 ])
            else:
                    sample_loss=0
                    uncertain_len = 1
            # sample_loss=sample_loss/len(text_x)if x

        print("APPENDING VALUES",sample_loss,params["uncertain_unlabelled"][ucounter])

        unlabelled_loss_list.append(beta_const*sample_loss+(1-beta_const)*params["uncertain_unlabelled"][ucounter])#
        # unlabelled_loss_list(np.sum(delta_delta_final)/len(text_x))
        ucounter+=1
    params["delta_dict"] = delta_dict
    # for sample in params["all_text_data"]:
    #     seq_len=len(sample.split())
    #     sample_loss=loss_bar[start_idx:start_idx+seq_len]
    #
    #     assert len(sample_loss)==seq_len,"not same length"
    #     start_idx=start_idx+seq_len
    #     unlabelled_loss_list.append(sample_loss)
    # print(start_idx+len(unlabelled_texts[-1].split()),len(loss_bar))
    # assert start_idx+len(unlabelled_texts[-1].split())==len(loss_bar)
    return np.array(unlabelled_loss_list)

def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags

def regression_parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    # tags = [0.1 for t in line_tags.strip().split()]
    if np.isscalar(line_tags):
        tags = [0.1 for t in words]
    else:
        tags = [t for t in line_tags]
    # tags=np.reshape(tags,len(words))
    # tags = line_tags
    # assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    return ((words, len(words)), (chars, lengths)), tags

def generator_fn(words, tags):
    '''words and tags are words_path and file_paths'''
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)



def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),  # (words, nwords)
               ([None, None], [None])),  # (chars, nchars)
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset

def selected_generator_fn(f_words, f_tags):
    f_words=np.array(f_words)#[selected_idx]
    f_tags=np.array(f_tags)#[selected_idx]
    for line_words, line_tags in zip(f_words, f_tags):
        yield parse_fn(line_words, line_tags)

def selected_regression_generator_fn(f_words, f_tags):
    f_words = np.array(f_words)  # [selected_idx]
    # f_tags = np.reshape(f_tags, (-1,1))  # [selected_idx]
    for line_words, line_tags in zip(f_words, f_tags):
        yield regression_parse_fn(line_words, line_tags)

def selected_input_fn(word_list, tag_list, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),  # (words, nwords)
               ([None, None], [None])),  # (chars, nchars)
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(selected_generator_fn, word_list, tag_list),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset
def selected_regression_input_fn(word_list, tag_list, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),  # (words, nwords)
               ([None, None], [None])),  # (chars, nchars)
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.float32)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                0.0)
    dataset = tf.data.Dataset.from_generator(
        functools.partial(selected_regression_generator_fn, word_list, tag_list),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['regression_epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset
def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,seed=0,
                                        training=training)

    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training,seed=0)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training,seed=0)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, seq_score, best_score,_ = crf.crf_decode(logits, crf_params, nwords)


    # seq_score=[x for x in seq_score]
    # pred_ids,best_score = crf.viterbi_decode(seq_score, crf_params)
    if mode == tf.estimator.ModeKeys.PREDICT:

        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        probabilities = tf.nn.softmax(logits, axis=-1)
        # seq_score2 = crf.crf_sequence_score(logits, pred_ids, nwords, crf_params)

        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings,
            'seq_score':seq_score,
            'best_score':best_score,
            'probabilities':probabilities
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)
        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        if mode == tf.estimator.ModeKeys.EVAL:

                tf.logging.info("***** Eval results *****")

                tf.logging.info( "*****************LOOP: " + str(params["loop_no"]) + "*****" + datetime.now().strftime(
                            "%m_%d_%Y_%H_%M_%S") + "*************\n")
                tf.logging.info("*****************Sentence count" + str(params["trained_len"]) + "/" + str(
                    params["all_data_len"]) + '=' + str(
                    params["trained_len"] / params["all_data_len"]*100) + "*************\n")
                tf.logging.info("token_count" + str(params["train_token_count"]) + "/" + str(
                    params["all_token_count"]) + '=' + str(
                    params["train_token_count"] / params["all_token_count"] * 100) + "*************\n")
                if loop_no>0:
                    # writer.write("*****************concept_count:" + str(params["trained_concept_count"])+"/"  + str(params["all_concept_count"])+'='+str(params["trained_concept_count"]/params["all_concept_count"]* 100)+ "*************\n")
                    # writer.write("token_count" + str(params["train_token_count"]) + "/" + str(
                    #         params["all_token_count"]) + '=' + str(
                    #         params["train_token_count"] / params["all_token_count"] * 100) + "*************\n")
                    tf.logging.info("*****************concept_count:" + str(params["trained_concept_count"])+"/"  + str(params["all_concept_count"])+'='+str(params["trained_concept_count"]/params["all_concept_count"]* 100)+ "*************\n")

        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])
            # tf.logging.info(metric_name,op[1],'\n')
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.compat.v1.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        # super function inherits from nn.Module so that we can access everything from nn.Module
        super(LinearRegression,self).__init__()
        # Linear function
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)

def regression_model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                        training=training)

    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])

    output = tf.layers.dropout(output, rate=dropout, training=training)

    # Linear
    # linear=tf.keras.layers.Dense(params['lstm_size'], activation='relu')
    # output=linear(output)
    hidden_size = output.shape[-1]
    batch_size=output.shape[0]
    # logits = tf.layers.dense(output, 1)
    output_weights = tf.get_variable(
        "output_weights", [1, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [1], initializer=tf.zeros_initializer())

    # output_layer=tf.reshape(output,
    #                                [params["batch_size"] *  (params['dim']+params["dim_chars"]), params['lstm_size']])
    #
    # logits = tf.matmul(output, output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    linear = tf.keras.layers.Dense(1, activation=tf.nn.elu)
    # logits = tf.reshape(logits,[batch_size,-1])
    logits = linear(output)
    pred_ids=logits


    # seq_score=[x for x in seq_score]
    # pred_ids,best_score = crf.viterbi_decode(seq_score, crf_params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred_ids': pred_ids,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        # loss_fn = F.mse_loss
        # if params==None:
        #     trained_idx=[x for x in range(100)]

        # loss = loss_fn(logits, labels)

        # labels=torch.tensor(labels)
        # labels=tf.reshape(labels,[batch_size,-1,1])
        logits=array_ops.squeeze(logits, axis=[2])
        loss=tf.reduce_mean(tf.square(logits - labels))

        # Metrics
        # weights = tf.sequence_mask(nwords)
        metrics = {
            # 'f1': tf.reduce_mean(tf.square(logits - labels)),
            # 'acc':tf.reduce_mean(tf.square(logits - labels)),
            # 'precision':tf.reduce_mean(tf.square(logits - labels)),
            # 'recall':tf.reduce_mean(tf.square(logits - labels))

        }
        # if mode == tf.estimator.ModeKeys.EVAL:
        #
        #         tf.logging.info("***** Eval results *****")
        #
        #         tf.logging.info( "*****************LOOP: " + str(params["loop_no"]) + "*****" + datetime.now().strftime(
        #                     "%m_%d_%Y_%H_%M_%S") + "*************\n")
        #         tf.logging.info("*****************Sentence count" + str(params["trained_len"]) + "/" + str(
        #             params["all_data_len"]) + '=' + str(
        #             params["trained_len"] / params["all_data_len"]*100) + "*************\n")
        #         tf.logging.info("token_count" + str(params["train_token_count"]) + "/" + str(
        #             params["all_token_count"]) + '=' + str(
        #             params["train_token_count"] / params["all_token_count"] * 100) + "*************\n")
                # if loop_no>0:
                    # writer.write("*****************concept_count:" + str(params["trained_concept_count"])+"/"  + str(params["all_concept_count"])+'='+str(params["trained_concept_count"]/params["all_concept_count"]* 100)+ "*************\n")
                    # writer.write("token_count" + str(params["train_token_count"]) + "/" + str(
                    #         params["all_token_count"]) + '=' + str(
                    #         params["train_token_count"] / params["all_token_count"] * 100) + "*************\n")
                    # tf.logging.info("*****************concept_count:" + str(params["trained_concept_count"])+"/"  + str(params["all_concept_count"])+'='+str(params["trained_concept_count"]/params["all_concept_count"]* 100)+ "*************\n")

        # for metric_name, op in metrics.items():
        #     tf.summary.scalar(metric_name, op[1])
            # tf.logging.info(metric_name,op[1],'\n')
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.compat.v1.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


def match_idx_with_root_data(top_k_indices,idx_map):
    # print ("idx data",top_k_indices)
    # print("idx_map",idx_map)
    return [idx_map[x] for x in top_k_indices]

def update_dict(idx_map,new_idx):
    idx_map[idx_map["count"]]=new_idx
    idx_map["count"]=idx_map["count"]+1
def map_apart_of_data(all_data,part_of_idx):
    idx_map = {"count": 0}
    '''GENERATE ARRAY FROM IDX 0 FOR ANOTHER PART OF DATA'''
    [update_dict(idx_map, i) for i, x in enumerate(all_data) if i not in part_of_idx]
    return idx_map


def do_train_eval(estimator,selected_words, selected_tags):
    # Params


    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    # train_inpf = functools.partial(input_fn, fwords('i2b2_train'), ftags('i2b2_train'),
    #                                params, shuffle_and_repeat=True)
    # eval_inpf = functools.partial(input_fn, fwords('i2b2_test'), ftags('i2b2_test'))

    train_inpf = functools.partial(selected_input_fn, selected_words, selected_tags,
                                   params,shuffle_and_repeat=True)

    eval_inpf = functools.partial(input_fn, fwords(params["data_name"]+'_test'), ftags(params["data_name"]+'_test'))
    hook = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator, 'f1',500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    # for i in range(20):
    tf.estimator.train_and_evaluate(estimator, train_spec,eval_spec)
    #

    # result = estimator.evaluate(eval_spec)



def do_regression_train_eval(reg_estimator,selected_words, selected_loss):
    # Params


    # def fwords(name):
    #     return str(Path(DATADIR, '{}.words.txt'.format(name)))
    #
    # def ftags(name):
    #     return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    # train_inpf = functools.partial(input_fn, fwords('i2b2_train'), ftags('i2b2_train'),
    #                                params, shuffle_and_repeat=True)
    # eval_inpf = functools.partial(input_fn, fwords('i2b2_test'), ftags('i2b2_test'))

    train_inpf = functools.partial(selected_regression_input_fn, selected_words, selected_loss,
                                   params,shuffle_and_repeat=True)
    eval_inpf = functools.partial(selected_regression_input_fn, selected_words, selected_loss)
    hook = tf.estimator.experimental.stop_if_no_increase_hook(
        reg_estimator, 'f1',500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)#, hooks=[hook]
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    # for i in range(20):
    tf.estimator.train_and_evaluate(reg_estimator, train_spec,eval_spec)
    return reg_estimator

'''@TODO: rewrite estimator.predict, not untrained data, function'''
def write_predictions(estimator,unlabelled_texts, unlabelled_tags):
    Path(output_dir + 'score').mkdir(parents=True, exist_ok=True)
    with Path(output_dir + 'score/{}.preds.txt'.format('results')).open('wb') as f:
        test_inpf = functools.partial(selected_input_fn, unlabelled_texts, unlabelled_tags)
        golds_gen = selected_generator_fn(unlabelled_texts,unlabelled_tags)
        preds_gen = estimator.predict(test_inpf)
        for golds, preds in zip(golds_gen, preds_gen):
            ((words, _), tags) = golds
            for word, tag, tag_pred in zip(words, tags, preds['tags']):
                f.write(b' '.join([word, tag, tag_pred]) + b'\n')
            f.write(b'\n')
def get_LC_score(result,text_len):
    # prob_seq = [1-float(x) for x in np.amax(result["seq_score"], axis=1)]
    prob_seq = [1-float(x) for x in np.amax(result["probabilities"], axis=1)][0:text_len]
    # final_score = 1 - np.prod(prob_seq)
    # final_score=1-prob_seq[-1]
    # final_score =np.sum(prob_seq)#/text_len
    final_score = np.sum(prob_seq)
    return final_score
def get_MNLP_score(result,text_len):
    # print(result["seq_score"])
    prob_seq = [ math.log1p(float(x)) if x!=0 else 0  for x in np.amax(result["seq_score"], axis=1)][:text_len]
    # final_score = np.sum(prob_seq)/len(prob_seq)
    final_score = -np.sum(prob_seq)/text_len
    # final_score= -[ math.log1p(float(x))   for x in np.amax(result["seq_score"], axis=1)]
    # final_score=-result["best_score"]/result["word_len"]
    return final_score
def get_LCC_score(result):
    prob_seq = [float(x) for x in np.amax(result["seq_score"], axis=1)]
    label_seq = [x for x in result["pred_ids"].flat]
    # print("\nPROB_SEQ",len(prob_seq),len(label_seq))
    start_chunk_idx, end_chunk_idx, max_pos = data_util.find_max_sequence(label_seq, use_i2b2)
    seq_confidence_score = [1 - np.average(prob_seq[start_seq:end_chunk_idx[i]]) for i, start_seq in
                            enumerate(start_chunk_idx)]
    final_score = np.amax(seq_confidence_score)
    return final_score
def select_next_training_data(estimator,unlabelled_texts, unlabelled_tags,top_k_num,params):
    limit_addition = int(params["plan_sentence_count"] if use_sentence_base else params["plan_token_count"])
    trained_indices = params["trained_idx"]
    simi_train_scores = file_util.load(
        simulation_folder + "/" + data_name + "simi_train_scores.pck"+str(params["test_neighbor_num"]))  # .toarray()
    simi_test_scores = file_util.load(
        simulation_folder + "/" + data_name + "simi_test_scores.pck"+str(params["test_neighbor_num"]))  # .toarray()
    simi_desired_thres_idx_dict=file_util.load(simulation_folder + "/"+data_name+"simi_desired_thres_idx.pck"+str(params["test_neighbor_num"]))

    # simi_test_vector = file_util.load(
    #     simulation_folder + "/" + data_name + "simi_test_vector.pck"+str(params["test_neighbor_num"]))
    # simi_test_idx = file_util.load(
    #     simulation_folder + "/" + data_name + "simi_diss_test_idx.pck"+str(params["test_neighbor_num"]))
    result_counter=0
    delta_dict=params["delta_dict"]
    test_inpf = functools.partial(selected_input_fn, unlabelled_texts, unlabelled_tags,params)

    results = estimator.predict(test_inpf, yield_single_examples=True)
    # tf.logging.info("TOKENS",params["plan_token_count"])
    result_counter=0
    word_len_list=[]
    result_data=[]

    id_to_tag_vocab=params["id_to_tag_vocab"]
    tag_to_id_vocab=params["tag_to_id_vocab"]
    simi_test_idx = file_util.load(
        simulation_folder + "/"+data_name+"simi_diss_test_idx.pck")

    # result_count=0
    # dissi_arrays = dissi_matrix.toarray()
    # istrained_arrays = np.zeros(len(all_texts_data))
    # istrained_arrays[params["trained_idx"]] = [1] * len(params["trained_idx"])
    # istrained_matrix=istrained_arrays*len(all_texts_data)
    # untrained_dissi_scores = np.multiply(dissi_arrays, istrained_matrix)[params["unlabelled_idx"]]
    # untrained_dissi_scores=[np.sum(x)/len(all_texts_data) for x in untrained_dissi_scores]
    # simi_dict = file_util.load(
    #     simulation_folder + "/simi_dict_" + str(dissi_neighbor_num) + "neighbor_" + data_name + ".pck")
    # for result in results:
    #     # print(result["log_probs"])
    #     # print("FLAT",result["log_probs"].flat)
    #     print("\nPREDICT UNLABELLED SAMPLES:",result_counter,'/',len(unlabelled_texts))
    #
    #
    #         # assert result_counter<len(silhouette_scores),"untrain_len:"+str(len(untrained_data))+'|'+str(result_counter)+'/'+str(len(silhouette_scores))+'|'+str(count)
    #     # if result_counter<len(silhouette_scores):
    #             # bald_context_score=bald*silhouette_scores[result_counter]
    #             # best_probs.append(bald_context_score)
    #     result["word_len"]=text_util.count_token([unlabelled_texts[result_counter]])
    #     word_len_list.append(result["word_len"])
    #     if use_LCC:
    #         context_score= get_LCC_score(result)#+silhouette_scores[result_counter]#word_entropy*#bald_seq
    #     elif use_LC:
    #         context_score=get_LC_score(result)
    #     elif use_MNLP:
    #         context_score=get_MNLP_score(result)
    #     best_probs.append(context_score)
    #     result_counter+=1
    #     prob_seq = [float(x) for x in np.amax(result["seq_score"], axis=1)]
    #     result_data.append({"tags":[x for x in result["tags"]],"seq_score":[float(x) for x in np.amax(result["seq_score"], axis=1)]})

    # assert len(best_probs)>0, 'results must have been > 0'
            # print("SCORE",bald_context_score)
    # best_probs = np.array(best_probs)
    # file_util.dump(result_data,output_dir+"/prob_tags.pck")
    # top_k_idx = best_probs.argsort() [-top_k_num:][::-1]
    sorted_len_idx = np.argsort([x for x in word_len_list])[::-1]
    # best_probs = best_probs[sorted_len_idx]
    '''BEGIN TOKEN ANALYSYS WITHOUT SORT'''
    trained_indices=params["trained_idx"]

    fscore_loss_list=params["fscore_loss"][trained_indices]
   # # trained_dissi_scores=np.multiply(dissi_arrays,istrained_arrays)

    #np.reshape(np.random.random((len(loss_list),1)),(-1,1)) #np.reshape(loss_list,(-1,1))

    neighbor_num = 30
    utrain_train_sim_diss_vector, utrain_train_simi_dissi_idx = text_util.get_simi_dissi_vector(
        params["unlabelled_idx"],
        simi_train_scores, trained_indices, neighbor_num)
    uncertain_test_instances = []
    test_lines = read_data(os.path.join(dataset_dir, "test.txt"))
    all_test_texts = np.array([x[1] for x in test_lines])
    portion_test_len=limit_addition
    all_test_tags = np.array([x[0] for x in test_lines])
    test_inpf = functools.partial(selected_input_fn, all_test_texts, all_test_tags, params)

    tresults = estimator.predict(test_inpf, yield_single_examples=True)
    result_counter = 0
    for result in tresults:
        test_text_len = len(all_test_texts[result_counter].split())
        pred_scores = [float(x) for x in np.amax(result["probabilities"], axis=1)][:test_text_len]
        # print("TRUE PRB",true_tag_probs
        # loss_i = np.array(
        #     [(true_tag_probs[j])*math.log1p(float(true_tag_probs[j])) if pred_tag[j] != y else 0 for j, y in enumerate(true_tag)])
        # loss_i = np.array(
        #     [(pred_scores[j]*fscore_loss_list[result_counter])  if pred_tag[j] != y else ((1-pred_scores[j])*fscore_loss_list[result_counter])  for j, y in enumerate(true_tag)])
        uncertain_i = get_LC_score(result,test_text_len)
        uncertain_test_instances.append(uncertain_i)
        result_counter += 1


    desired_simi_thres_counter=0

    # u_inpf = functools.partial(selected_input_fn, unlabelled_texts, unlabelled_tags, params)
    # uresults = estimator.predict(u_inpf, yield_single_examples=True)
    ucounter=0
    simi_thres = params["simi_thres_list"][params["simi_thres_id"]]

    related_test_data=params.get("related_test_idx", [])
    start_time=time.time()
    print("BEGIN TIME",start_time)
    # simi_test_instances = {u_target_id: [[z,u_target_id,simi_test_scores[u_target_id, z]] for z in np.where(simi_test_scores[u_target_id] >= simi_thres)[0] if z not in related_test_data] for u_target_id in params["unlabelled_idx"]}
    no_neighbors=params.get("no_neighbors",False)
    if no_neighbors==False :
        simi_test_instances = {u_target_id: [[z, u_target_id, simi_test_scores[u_target_id, z]] for z in
                                             simi_desired_thres_idx_dict[simi_thres][u_target_id] if
                                             z not in related_test_data] for u_target_id in params["unlabelled_idx"]}

        simi_test_instance_idx = {u_target_id: [x[0] for x in v] for u_target_id, v in simi_test_instances.items()}

        print("END TIME",time.time()-start_time)

        test_values = list(chain(*simi_test_instances.values()))

        simi_i = [x[0] for x in test_values]
        simi_test_num = len(list(set(simi_i)))

        print("SIMI TEST NUM", simi_test_num,simi_thres,portion_test_len)
    else:
        simi_test_instances = {u_target_id: [] for u_target_id in params["unlabelled_idx"]}

        simi_test_instance_idx = {u_target_id: [] for u_target_id, v in simi_test_instances.items()}

        print("END TIME", time.time() - start_time)

        test_values = None

        simi_i = None
        simi_test_num = 0

        print("SIMI TEST NUM", simi_test_num, simi_thres, portion_test_len)

    while(simi_test_num< portion_test_len and no_neighbors==False):
        if  params["simi_thres_id"]<len(params["simi_thres_list"])-1:
            params["simi_thres_id"] += 1
            simi_thres = params["simi_thres_list"][params["simi_thres_id"]]

            simi_test_instances = {u_target_id: [[z,u_target_id,simi_test_scores[u_target_id, z]] for z in simi_desired_thres_idx_dict[simi_thres][u_target_id] if z not in related_test_data] for u_target_id in params["unlabelled_idx"]}
            simi_test_instance_idx={u_target_id:[x[0] for x in v] for u_target_id,v in simi_test_instances.items()}
            test_values = list(chain(*simi_test_instances.values()))

            simi_i = [x[0] for x in test_values ]
            simi_test_num = len(list(set(simi_i)))
            print("SIMI TEST NUM",simi_test_num,simi_thres,portion_test_len)
        else:
            break
    usimi_desired_dict = {}
    if simi_test_num>0 and no_neighbors==False:
        simi_test_dict={test_id:{"idx":[],"simi_values":[],"uncertain_value":uncertain_test_instances[test_id]} for test_id in simi_i}
        """BEGIN FIND BEST PAIR"""
        # considered_test_idx = list(set(simi_i))

        print("process considered_test_idx")

        for test_val in test_values:
            test_id=test_val[0]
            simi_test_dict[test_id]["idx"].append(test_val[1])
            simi_test_dict[test_id]["simi_values"].append(test_val[2])



        print("process best fit of test_id")
        for test_id, v in simi_test_dict.items():
            best_utarget = v["idx"][np.argmax(v["simi_values"])]
            usimi_desired_dict[best_utarget] = usimi_desired_dict.get(best_utarget, [])
            usimi_desired_dict[best_utarget].append(test_id)
    """END FIND BEST PAIR"""
    uncertain_unlabelled_neighbors = []
    '''BEGIN UNLABELLED ESTIMATION'''
    for u_target_id in params["unlabelled_idx"]:
        # u_target_id=params["unlabelled_idx"][ucounter]
        # simi_train_test_id = simi_test_idx[u_target_id][0]
        # diss_train_test_id = simi_test_idx[u_target_id][1]
        len_text=len(unlabelled_texts[ucounter].split())

        # if len(simi_train_test_id) > 0:
        #     #simi_sum=uncertain_test_instances[simi_train_test_id[-1]]*simi_test_vector[u_target_id][:neighbor_num][-1]
        #     simi_sum = np.sum(uncertain_test_instances[simi_train_test_id[-1]]) * np.sum(simi_test_vector[u_target_id][:neighbor_num])
        #     # simi_sum=np.amax(np.array(uncertain_test_instances)[simi_train_test_id])
        # else:
        #     simi_sum=pow(math.e,-10)
        # desired_sim_id = [x for u, x in enumerate(simi_train_test_id) if
        #                   simi_test_vector[u_target_id][:neighbor_num][u] >= simi_thres and x not in params.get("related_test_idx",[])]
        # desired_sim_i = [u for u, x in enumerate(simi_train_test_id) if
        #                  simi_test_vector[u_target_id][:neighbor_num][u] >= simi_thres and x not in params.get("related_test_idx",[])]
        desired_sim_id=usimi_desired_dict.get(u_target_id,[])

        # desired_sim_i=[u for u, x in enumerate(simi_test_scores) if x in desired_sim_id]

        if len(desired_sim_id) > 0:
            # desired_sim_id =simi_test_instances[u_target_id]
            #
            # desired_sim_i=[u for u, x in enumerate(simi_train_test_id) if x in desired_sim_id]

            desired_simi_thres_counter+=1
            simi_arr=simi_test_scores[u_target_id][desired_sim_id]
            uncertain_arr=np.array(uncertain_test_instances)[desired_sim_id]

            # total_val = np.multiply(simi_arr,uncertain_arr)
            total_val = uncertain_arr
            total_val = np.amax(total_val)#+len(desired_sim_id)/portion_test_len
            total_val =undefined_value if total_val<uncertain_threshold else total_val#*len_text
            # total_val=
        else:
            total_val = undefined_value
        # u_uncertain_val=get_LC_score(result,len_text)

        if total_val!=undefined_value:
            print("usample:", ucounter,total_val)

        uncertain_unlabelled_neighbors.append(total_val)#
        ucounter+=1
    reduce_thres_level=len([x for x in uncertain_unlabelled_neighbors if x!=undefined_value])
    if reduce_thres_level<portion_test_len:
        if  params["simi_thres_id"] < len(params["simi_thres_list"]) - 1:
            params["simi_thres_id"] += 1
            params["no_neighbors"] = False
        else:
            params["no_neighbors"]=True
    '''END UNLABELLED ESTIMATION'''
    '''LOSS TRAINING'''
    ltest_inpf = functools.partial(selected_input_fn, params["labelled_texts"], params["labelled_tags"], params)

    lresults = estimator.predict(ltest_inpf, yield_single_examples=True)
    result_counter = 0
    loss_list = []
    result_counter=0
    print("process loss of labelled")
    for result in lresults:
        true_tag = params["labelled_tags"][result_counter].strip().split()
        true_tag_ids=[tag_to_id_vocab[x] for x in true_tag]

        scores=result["seq_score"][0][0]
        pred_tag = [id_to_tag_vocab[x] for x in result["pred_ids"].flat]
        # pred_scores =[ math.log1p(float(x))*x   for x in np.amax(xc, axis=1)]
        pred_scores = [float(x) for x in np.amax(result["probabilities"], axis=1)]
        seq_scores1=result["seq_score"]
        true_tag_probs=[result["probabilities"][i][x]  for i,x in enumerate(true_tag_ids)]
        # print("TRUE PRB",true_tag_probs
        # loss_i = np.array(
        #     [(true_tag_probs[j])*math.log1p(float(true_tag_probs[j])) if pred_tag[j] != y else 0 for j, y in enumerate(true_tag)])
        # loss_i = np.array(
        #     [(pred_scores[j]*fscore_loss_list[result_counter])  if pred_tag[j] != y else ((1-pred_scores[j])*fscore_loss_list[result_counter])  for j, y in enumerate(true_tag)])

        # loss_i = np.array(
        #     [math.log1p(pred_scores[j]) if pred_tag[j] != y else undefined_value  for j, y in enumerate(true_tag)])#if pred_tag[j] != y else 1-pred_scores[j]

        loss_i = np.array(
            [pred_scores[j] if pred_tag[j] != y else undefined_value for j, y in enumerate(true_tag)])
        # loss_i = np.array(
        #     [pred_scores[j] if pred_tag[j] != y else 0 for j, y in enumerate(true_tag)])

        # loss_i = np.array([-pred_scores[j] for j, y in  enumerate(true_tag)])



        last_loss=delta_dict[trained_indices[result_counter]][0]
        last_delta=delta_dict[trained_indices[result_counter]][1]
        delta_loss_i=alpha_const* loss_i+(1-alpha_const)*(loss_i-last_loss)
        current_delta_loss=loss_i-last_loss
        current_delta_delta=current_delta_loss-last_delta

        current_delta_loss=[x if x>0 else 0 for x in delta_loss_i]
        delta_delta_final=np.multiply(current_delta_loss,current_delta_delta)
        # delta_delta_final=[x if x >0 else 0 for x in delta_delta_final ]
        # delta_delta_final =0.7*np.array([x if x>0 else 0 for x in loss_i-last_loss]) +0.3*np.array([x if x > 0 else 0 for x in delta_delta_final])
        # delta_delta_final = np.array([x  for x in loss_i - last_loss])#if x > 0 else 0

        # if use_positive_loss:
        #     delta_loss_i=[x if x >0 else 0 for x in delta_loss_i]
        # else:
        #     delta_loss_i = [x if x <0 else 0 for x in delta_loss_i]

        delta_dict[trained_indices[result_counter]]=[loss_i,loss_i-last_loss]
        # loss_list.append(delta_delta_final)
        loss_list.append(loss_i)
        result_counter+=1
    params["rank_labels"] = loss_list
    #
    '''LOSS TRAINING'''
    #  all_inpf = functools.partial(selected_input_fn, params["all_text_data"],params["all_tag_data"], params)
   #  all_results = estimator.predict(all_inpf, yield_single_examples=True)
   #  dict_score_chunks=[]
   #  resulta_counter=0
   #  reg_simi=[]
   #  simi_uncertains=[]
   #  for result in all_results:
   #      certain_score_dict = data_util.find_chunk_score(params["tag_list"], result["seq_score"].flat,result["pred_ids"].flat, params["id_to_tag_vocab"])
   #      dict_score_chunks.append( {key: -value if value > 0 else 0 for key, value in certain_score_dict.items()})  # uncertain score
   #      label_uncertain=[certain_score_dict[y] for y in params["tags_wth_prefix"]]
   #      reg_simi.append(label_uncertain)
   #      simi_uncertains.append(get_MNLP_score(result))
   #      resulta_counter += 1
   #  print("all_results",resulta_counter)
   #  for i in range(params["all_data_len"]):
   #      simi_neighbors=simi_dict[i][:params["simi_num"]]
   #      tema=list(set(simi_neighbors).intersection(set(params["trained_idx"])))
   #      uu=[simi_uncertains[x] if x in tema else 0 for x in simi_dict[i]]
   #      reg_simi[i].extend(uu)
   #      print("extend",i)
   #  tags_wth_prefix = params["tags_wth_prefix"]
   #  score_chunk_array = [[x[y] for y in tags_wth_prefix] for x in dict_score_chunks]
   #  uncertain_chunk_csc = data_util.convert2csc3(score_chunk_array)
    # params["uncertain_matrix"] = uncertain_chunk_csc
    # uX=np.array(reg_simi[params["unlabelled_idx"]])

    # graph_scores = gcn_util.train_het(params)
    params["delta_dict"] = delta_dict
    print("predict unlabelled data")
    params["uncertain_unlabelled"] = uncertain_unlabelled_neighbors
    params["encoded_labelled_tags"]= [[z.encode() for z in t.strip().split()] for t in trained_tags]
    graph_scores=train_regression_loss(params)

    best_probs=graph_scores[:len(params["unlabelled_texts"])]
    # best_probs=np.array(uncertain_unlabelled_neighbors)
    # params["reg_X"]=np.array(reg_simi)[params["unlabelled_idx"]]
    # params["reg_U"]=np.array(reg_simi)[params["trained_idx"]]
    # graph_scores=regression_util.train_data(params)
    # best_probs=np.array([x+graph_scores[t] for t,x in enumerate(best_probs)])
    # print("LEN PROB",len(best_probs),len(graph_scores))
    # best_probs = np.array([untrained_dissi_scores[t] for t, x in enumerate(best_probs)])
    # best_probs = np.array([x + graph_scores[t]+untrained_dissi_scores[t] for t, x in enumerate(best_probs)])
    # best_probs=np.array([x[0] for x in best_probs])
    top_k_idx = best_probs.argsort()[-len(unlabelled_texts):][::-1][0:limit_addition]


    # print("BEST PROB",best_probs[3])
    a=params.get("related_test_idx",[])
    b=[]
    [b.extend(v) for k,v in usimi_desired_dict.items() if k in np.array(params["unlabelled_idx"])[top_k_idx]]
    a.extend(list(set(b)))
    params["related_test_idx"]=a
    return top_k_idx



def get_labels(self):
        # return ["B", "I", "O", "X", "[CLS]", "[SEP]"]
    if use_i2b2:
        return [ "B-treatment", "I-treatment",  "I-problem", "B-problem", "B-test", "I-test", "O","X", "[CLS]",
                "[SEP]","[PAD]"]#,
    else:
        return ["B-disorder", "I-disorder", "O", "X", "[CLS]", "[SEP]","[PAD]"]  #



def read_data( input_file):
        """Read a BIO data!"""
        def process_line(labels, words):
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            lines.append((l, w))
            words = []
            labels = []
            return words,labels,lines
        rf = open(input_file, 'r')
        lines = [];
        words = [];
        labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if not use_i2b2:
                if len(line.strip()) == 0:  # and words[-1] == '.'
                    words, labels, lines = process_line(labels, words)
            else:
                if len(line.strip()) == 0 :#and words[-1] == '.'
                    words, labels, lines = process_line(labels, words)
            words.append(word)
            labels.append(label)
        rf.close()
        return lines
if __name__ == '__main__':
    lines = read_data(os.path.join(dataset_dir, "train.txt"))
    test_lines = read_data(os.path.join(dataset_dir, "test.txt"))
    all_texts_data = np.array([x[1] for x in lines])
    all_tags_data = np.array([x[0] for x in lines])
    test_texts = np.array([x[1] for x in test_lines])

    single_token_idx = [i for i,x in enumerate(all_texts_data) if len(x) == 1 or len(x) == 2]
    all_token_count = text_util.count_token(all_texts_data)
    all_data_len = len(all_texts_data)
    #1 more step to convert tokens in 1 line to line
    params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 1,
        'regression_epochs':10,
        'batch_size': 16,
        'buffer': 15000,
        'filters': 25,
        'kernel_size': 3,
        'lstm_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz')),
        'dissi_num': int(0.45*all_data_len),
        'simi_num': int(0.45*all_data_len),
        'data_name': data_name,
        'text_no_tag_count':0,
        'test_neighbor_num':30,
        'simi_thres_id': 0,
        # 'simi_thres_list': [0.5],
        'simi_thres_list': [1, 0.9, 0.8, 0.7, 0.6, 0.5,0.4,0.3,0.2,0.1]
        # 'simi_thres_list': [1, 0.9, 0.8, 0.7, 0.6, 0.5],
    }



    with Path(output_dir + 'params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120,tf_random_seed=12)
    percent_list=[0.73,0.19, 0.77, 0.96, 1.92, 3.84, 3.84, 3.84, 3.84, 9.61, 9.6, 9.6, 9.6, 19.21, 19.2,5]
    percent_list=[1]*ite_num
    # percent_list[0]=0.73
    # percent_list = [0.05, 1.64, 4.3, 7.63, 9.77, 12.23, 15.05, 17.65, 19.72, 24.69, 29.49, 34.84, 40.29, 54.65, 87.23,
    #                 100]
    # percent_list =[5]*21

    # percent_list = [x - percent_list[i - 1] if i > 0 else x for i, x in enumerate(percent_list)]
    al_sentence_amount_list=[101,141,299,454,748,1328,1859,2390,2933,4105]
    delta_dict={t: [np.zeros(len(all_texts_data[t].split())),np.zeros(len(all_texts_data[t].split()))] for t,x in enumerate(all_texts_data)}
    params["delta_dict"]=delta_dict

    if os.path.exists(
            simulation_folder + "/"+data_name+"simi_test_scores.pck"+str(params["test_neighbor_num"])):
        pass
    else:
        simi_train_scores, simi_test_scores, simi_desired_thres_idx = text_util.create_simi_matrix_train_test(
            all_texts_data, test_texts, neighbor_num=params["test_neighbor_num"],using_bert=False)

        file_util.dump(simi_test_scores,
                       simulation_folder + "/"+data_name+"simi_test_scores.pck"+str(params["test_neighbor_num"]))
        file_util.dump(simi_train_scores,
                       simulation_folder + "/"+data_name+"simi_train_scores.pck"+str(params["test_neighbor_num"]))
        file_util.dump(simi_desired_thres_idx,
                       simulation_folder + "/"+data_name+"simi_desired_thres_idx.pck"+str(params["test_neighbor_num"]))

    # run_config = tf.contrib.tpu.RunConfig(
    #     cluster=None,
    #     master=None,
    #     model_dir=output_dir,
    #     save_checkpoints_steps=1000,
    #     tpu_config=tf.contrib.tpu.TPUConfig(
    #         iterations_per_loop=1000,
    #         num_shards=8,
    #         num_shards=8,
    #         per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
    # estimator = tf.contrib.tpu.TPUEstimator(
    #     use_tpu=False,
    #     model_fn=model_fn,
    #     config=run_config,
    #     train_batch_size=24,
    #     eval_batch_size=24,
    #     predict_batch_size=24)

    # if use_i2b2:
    #     seed_pct=0.005
    # else:
    #     seed_pct=0.01

    # tf.random.set_seed(1234)

    params["all_data_len"]=all_data_len
    simulation_folder = "/home/s4616573/data/bert_ner_output/data_simulation_al_gcn/"
    dissi_neighbor_num = params["dissi_num"]
    simi_neighbor_num = params["simi_num"]
    data_name = params["data_name"]
    trained_matrix = np.zeros(shape=(all_data_len, all_data_len))
    tags = [re.sub('\n', '', str(x)) for x in open(str(Path(DATADIR, 'vocab.tags.txt'))).readlines()]
    # tags=[x for x in tags  if x!='O']
    id_to_tag_vocab = {i: x for i, x in enumerate(tags)}
    tag_to_id_vocab = {x: i for i, x in enumerate(tags)}
    params["id_to_tag_vocab"] = id_to_tag_vocab
    params["tag_to_id_vocab"] = tag_to_id_vocab
    params["tag_list"] = [x for x in tags if x != 'O']
    tags_wth_prefix = [x.split('-')[-1] for x in tags]
    tags_wth_prefix = list(set([x for x in tags_wth_prefix if x != 'O']))
    params["tags_wth_prefix"] = tags_wth_prefix
    params["fscore_loss"]=np.zeros((all_data_len))

    # if not os.path.exists(
    #         simulation_folder + "/simi_seeds.pck"):
    #     simi_count_dict = {key: 0 for key, value in simi_dict.items()}
    #     for simi_idx, simi_arr in simi_dict.items():
    #         for simi_e in simi_arr:
    #             simi_count_dict[simi_e] += 1
    #     simi_seed = []
    #     for key in sorted(simi_count_dict, reverse=True):
    #         simi_seed.append(key)
    #     simi_seed = simi_seed[:simi_neighbor_num]
    #     file_util.dump(simi_seed, simulation_folder + "/simi_seeds.pck")
    # else:
    #     simi_seed=file_util.load(simulation_folder + "/simi_seeds.pck")
    for i,percent_value in enumerate(percent_list):
        params["loop_no"]=i
        # paramss["pc_no"]=percent_list

        loop_no=i
        params["all_token_count"] = all_token_count
        if i==0:
            # unlike_pct = batch_sample_size
            # seeds = generate_seed(all_texts_data, seed_pct)
            # seed_idx = [x["corpus_id"] for x in seeds]
            # seed_len=int(len(all_texts_data)*percent_list[0]/100)
            plan_token_count=all_token_count*percent_list[0]/100
            plan_sentence_count=int(math.ceil(len(all_texts_data)*percent_list[0]/100))
            # seed_idx =[x for x in  range(0,plan_sentence_count,1)]
            if seed_set_num!=-1:
                seed_idx = file_util.load(
                    "/home/s4616573/data/bert_ner_output/data_simulation_al_gcn/" + data_name + ".seeds." + str(
                        seed_set_num) + ".pck")
            else:
                seed_idx = [x for x in range(0, plan_sentence_count, 1)]
            print("SEED",seed_idx)
            # seed_tokens=0
            # seed_idx = simi_seed[:plan_sentence_count]

            # sortedindex = np.argsort([len(x) for x in all_texts_data])[::-1]
            # # sortedindex = [x for x in range(len(all_texts_data))]
            # for s_idx in sortedindex:
            #     sample_len=text_util.count_token([all_texts_data[s_idx]])
            #     seed_tokens+=sample_len
            #     if seed_tokens<plan_token_count:
            #     # if len(seed_idx)<102:
            #         seed_idx.append(s_idx)
            #     else:
            #         break


            file_util.dump(seed_idx, os.path.join(output_dir, "trained_indices.pck"))
            file_util.dump(seed_idx, os.path.join(output_dir, "trained_indices_loop0.pck"))
            selected_texts=[x for i,x in enumerate(all_texts_data) if i in seed_idx]
            selected_tags = [x for i, x in enumerate(all_tags_data) if i in seed_idx]
            trained_indices = seed_idx
            selected_data = [x for i, x in enumerate(all_texts_data) if i in seed_idx]
            train_token_count = text_util.count_token(selected_texts)
            params["train_token_count"] = train_token_count
            params["latest_selected_idx"] = seed_idx
        else:#COMBINE WITH BERT_RESULTS
            # if i>10:
            #     params['regression_epochs']=50
            trained_indices=file_util.load(os.path.join(output_dir, "trained_indices.pck"))
            # unlike_pct = int(percent_list[i]/100)
            # print("unlike_Pct",unlike_pct)
            # trained_indices = cikm.seed_idx
            '''WITHOUT BALD'''
            # root_map_idx=map_apart_of_data(all_training_data,trained_indices)
            # selected_data=generate_seed(untrained_data,unlike_pct,is_seed=False)
            # selected_idx = [x["id"] for x in selected_data]
            # selected_mapped_root_idx=match_idx_with_root_data(selected_idx,root_map_idx)
            # selected_texts=[x for i,x in enumerate(all_training_data) if i in selected_mapped_root_idx]
            '''END WITHOUT BALD'''
            '''WITH BALD'''
            root_map_idx=map_apart_of_data(all_texts_data,trained_indices)
            trained_texts=np.array(all_texts_data)[trained_indices]
            trained_tags = np.array(all_tags_data)[trained_indices]
            untrained_texts =[x for i_x,x in enumerate(all_texts_data) if i_x not in trained_indices ]
            untrained_tags = [x for i_x,x in enumerate(all_tags_data) if i_x not in trained_indices ]
            # trained_semtypes,namemention_dict=semtype_util.get_semtype_defs(trained_texts,use_i2b2)
            untrained_semtype_list=[]
            # for text in untrained_texts:
            #     untrained_semtypes, _ = semtype_util.get_semtype_defs([text],use_i2b2)
            #     untrained_semtype_list.append(untrained_semtypes)
            # clustered_scores0=cluster_util.do_semtype_cluster(trained_semtypes,untrained_semtype_list)
            # clustered_scores1=cluster_util.do_cluster(trained_texts,untrained_texts)
            # clustered_scores=[clustered_scores0[i]+clustered_scores1[i] for i,x in enumerate(clustered_scores0)]

            # lssv_scores,semtype_count,concept_lines=semtype_util.get_lssv_scores(untrained_texts,namemention_dict,len(trained_semtypes))
            # formatted_untrained_data=[x for i,x in enumerate(all_formatted_training_data) if i not in trained_indices]
            # print("LEN UNTRAINED DATA+ CLUSTER",len(formatted_untrained_data),len(clustered_scores))
            # selected_idx = cikm.select_kholgi_next_training_data(formatted_untrained_data,unlike_pct,lssv_scores)
            params["plan_token_count"] = round(percent_list[i] * all_token_count / 100, 2)
            params["plan_sentence_count"]=round(percent_list[i] * all_data_len / 100, 2)
            # params["plan_sentence_count"] = al_sentence_amount_list[i]#round(percent_list[i] * len(all_texts_data) / 100, 2)
            # tf.logging.info("PLAN TOKEN COUNT",plan_token_count)
            # for tid in trained_indices:
            #     trained_rows = [tid] * len(dissi_dict[tid])
            #     trained_matrix[trained_rows, dissi_dict[tid]] = 1
            params["all_text_data"]=all_texts_data
            params["all_tag_data"]=all_tags_data
            params["trained_matrix"] = sparse.csr_matrix(trained_matrix)
            params["unlabelled_texts"] = untrained_texts
            params["unlabelled_tags"] = untrained_tags
            params["labelled_tags"] = trained_tags
            params["labelled_texts"] = trained_texts
            params["trained_idx"] = trained_indices
            params["unlabelled_idx"]=[u for u, x in enumerate(all_texts_data) if u not in trained_indices]
            selected_idx = select_next_training_data(estimator,untrained_texts,untrained_tags, 0,params)

            assert len(selected_idx)>0,"Must be >0"
            selected_mapped_root_idx=match_idx_with_root_data(selected_idx,root_map_idx)

            # selected_mapped_root_idx.extend(trained_indices)#ADD TRAINED DATA OF LAST LOOP
            # selected_data=[x for i,x in enumerate(all_formatted_training_data) if i in selected_mapped_root_idx]
            selected_texts = all_texts_data[selected_mapped_root_idx]
            selected_tags = all_tags_data[selected_mapped_root_idx]
            # selected_concept_lines=np.array([x for i, x in enumerate(concept_lines) if i in selected_idx]).flatten().tolist()
            # trained_concepts=np.array([v["cuis"] for k,v in namemention_dict.items() ]).flatten().tolist()
            all_used_concepts=0#selected_concept_lines
            # all_used_concepts.extend(trained_concepts)
            # all_used_concepts=list(set(all_used_concepts))
            # concept_count=len(all_used_concepts)

            params["trained_concept_count"] = 0#concept_count
            params["all_concept_count"] =1# len(concept_lines)
            # file_util.dump(selected_idx, output_dir + "selected_idx_" + str(params["loop_no"]) + ".pck")
            file_util.dump(selected_texts,output_dir + "selected_text_" + str(params["loop_no"]) + ".pck")
            file_util.dump(selected_mapped_root_idx, output_dir + "selected_idx_" + str(params["loop_no"]) + ".pck")
            params["latest_selected_idx"]=selected_mapped_root_idx
            trained_indices.extend(selected_mapped_root_idx)
            contains_duplicates = any(trained_indices.count(element) > 1 for element in trained_indices)
            assert contains_duplicates==0, "DUPLICATE"


            print(contains_duplicates)
            train_token_count = text_util.count_token(all_texts_data[trained_indices])
            params["train_token_count"] = train_token_count
            file_util.dump(trained_indices, os.path.join(output_dir, "trained_indices.pck"))
            '''END BALD'''
            '''MAP SELECTED SEEDS TO ROOT IDX'''


        params["trained_len"] = len(trained_indices)
        # print(len(selected_texts))
        # if i>=loop_range-1:
        #     cikm.do_train_fuction(selected_data,True)
        # else:
        #     cikm.do_train_fuction(selected_data)
        print("DO TRAIN")
        estimator = tf.estimator.Estimator(model_fn, output_dir, cfg, params)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        hook = tf.estimator.experimental.stop_if_no_increase_hook(
            estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        # print("trained_concept_count", params["trained_concept_count"])
        # do_train_eval(estimator,selected_texts, selected_tags)
        do_train_eval(estimator, all_texts_data[trained_indices], all_tags_data[trained_indices])
        all_fscores=get_fscores()
        params["fscore_loss"][params["latest_selected_idx"]]=np.full((len(params["latest_selected_idx"])),all_fscores[-1]-all_fscores[-2]) if len(all_fscores)>1 else np.full((len(params["latest_selected_idx"])),all_fscores[-1])
        # params["labelled_texts"]=all_texts_data[:102]
        # params["labelled_tags"] = all_tags_data[:102]
        # params["rank_labels"]=np.reshape(np.random.random((102,1)),(-1,1))
        # print(params["rank_labels"].shape)
        # train_regression_loss(params)







