"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
import os
import utils.file_util as file_util
import utils.data_util as data_util
import utils.semtype_util as semtype_util
import utils.text_util as text_util
import utils.cluster_util as cluster_util
import utils.crf as crf
import shutil
import datetime
from datetime import datetime
import math
from masked_conv import masked_conv1d_and_max
import collections

tf.enable_eager_execution()
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

DATADIR = '../../data/example'
DATADIR = 'data/example'
data_name='i2b2'
seed_set_num=8
params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 1,
        'batch_size': 16,
        'buffer': 15000,
        'filters': 25,
        'kernel_size': 3,
        'lstm_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz')),
        'data_name':data_name
    }
method_name='MNLP'
use_LC=True if method_name=='LC' else False
use_MNLP=True if method_name=='MNLP' else False

# Logging

use_sentence_base=True
output_dir="/home/s4616573/data/bert_ner_output/al_"+params['data_name']+'_'+method_name+"/"
# logging.basicConfig(filename=output_dir+"logging_al.txt",filemode='a',level=logging.INFO)

use_i2b2=True if data_name=='i2b2' else False
if use_i2b2:
    dataset_dir="/home/s4616573/data/i2b2/"
else:
    dataset_dir="/home/s4616573/data/CLEF/"
use_conll=True if data_name=='conll' else False
if use_conll:
    dataset_dir = "/home/s4616573/data/conll-corpora/conll2003/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    shutil.rmtree(output_dir,ignore_errors=True)
    os.mkdir(output_dir)
Path(output_dir+'results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler(output_dir+'results/'+params['data_name']+'_'+method_name+'_seed_'+str(seed_set_num)+'.100ite.txt'),
    logging.StreamHandler(sys.stdout)
]
# tf.nn.sigmoid_cross_entropy_with_logits
logging.getLogger('tensorflow').handlers = handlers

def sort_score_by_len1(best_probs,word_len_list,sorted_len_idx,current_token_sum,token_total,all_idx,start_idx):
    # sorted_len_idx = np.argsort([x for x in word_len_list])[::-1]

    consider_sorted_idx=sorted_len_idx[start_idx:start_idx+10]
    consider_probs=best_probs[consider_sorted_idx]
    prob_mapping={i : w for i, w in enumerate(consider_sorted_idx)}
    len_sorted_prob_idx=consider_probs.argsort()[-len(consider_sorted_idx):][::-1]
    print(len_sorted_prob_idx,prob_mapping)
    top_10_idx=match_idx_with_root_data(len_sorted_prob_idx,prob_mapping)
    for idx in top_10_idx:
        if current_token_sum<token_total:
            all_idx.append(idx)
            current_token_sum+=word_len_list[idx]
        else:
            break
    if current_token_sum<token_total:
        sort_score_by_len1(best_probs, word_len_list, sorted_len_idx,  current_token_sum, token_total,all_idx, start_idx+10)

def sort_score_by_len2(word_len_list, prob_sorted_idx, current_token_sum, token_total, all_idx,
                           start_idx,step=10):
    try:
        # sorted_len_idx = np.argsort([x for x in word_len_list])[::-1]
        print("start idx", start_idx)
        # prob_sorted_idx = best_probs.argsort()[-len(best_probs):][::-1]
        consider_sorted_idx = prob_sorted_idx[start_idx:start_idx + step]
        consider_len = word_len_list[consider_sorted_idx]
        prob_mapping = {i: w for i, w in enumerate(consider_sorted_idx)}
        len_sorted_prob_idx = consider_len.argsort()[-len(consider_len):][::-1]
        #
        top_10_idx = match_idx_with_root_data(len_sorted_prob_idx, prob_mapping)
        for idx in top_10_idx:
            if current_token_sum < token_total:
                all_idx.append(idx)
                if not use_sentence_base:
                    current_token_sum += word_len_list[idx]
                else:
                    current_token_sum += 1
            else:
                break
        # if current_token_sum < token_total and len(prob_sorted_idx[start_idx:])>3:
        if current_token_sum < token_total and len(prob_sorted_idx[start_idx:]) > 0:
            sort_score_by_len2(word_len_list, prob_sorted_idx, current_token_sum, token_total, all_idx,
                               start_idx + step, step)
        else:
            return
    except:
        print("An exception occurred",start_idx,)
        raise

    # top_k_idx = best_probs.argsort()[-len(unlabelled_texts):][::-1]


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

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, seq_score, best_score,best_score_log = crf.crf_decode(logits, crf_params, nwords)
    # seq_score=[x for x in seq_score]
    probabilities = tf.nn.softmax(logits, axis=-1)
    # pred_ids,best_score = crf.viterbi_decode(seq_score, crf_params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings,
            'seq_score':seq_score,
            'best_score':best_score,
            'best_score_log': best_score_log,
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
def generate_seed(corpus,pct_number,is_seed=True):
    # print("corpus0", corpus[0])
    # corpus_texts=[tokenize.untokenize(x) for x in corpus]

    cluster_dict, highest_cluster= cluster_util.kmean_clustering(corpus,is_seed)
    highest_cluster_info=cluster_dict[highest_cluster["cluster_num"]]
    # print("CLUSTER_INFO",highest_cluster_info.keys())
    seeds=[]
    for cluster_data in highest_cluster_info["details"]:
            # {"cluster_id": cluster_id, "score": sample_silhouette_values[sentence_id],
            #  "text": corpus[sentence_id], "id": sentence_id})

        seed_len=int(pct_number*len(cluster_data))
        seeds.extend(cluster_data[0:seed_len])
    return seeds


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

    eval_inpf = functools.partial(input_fn, fwords(params['data_name']+'_test'), ftags(params['data_name']+'_test'))
    hook = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator, 'f1',500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    # for i in range(20):
    tf.estimator.train_and_evaluate(estimator, train_spec,eval_spec)
    #

    # result = estimator.evaluate(eval_spec)


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
def get_LC_score(result):
    # prob_seq = [1-float(x) for x in np.amax(result["seq_score"], axis=1)]
    prob_seq = [float(x) for x in np.amax(result["probabilities"], axis=1)][:result["word_len"]]
    # prob_seq = [1-float(x) for x in np.amax(result["probabilities"], axis=1)][:result["word_len"]]
    # final_score=1-np.amax(prob_seq)
    # final_score = 1 - np.prod(prob_seq)
    # final_score=1-prob_seq[-1]
    final_score=np.sum(prob_seq)#/result["word_len"]
    # final_score = np.sum(prob_seq)
    return final_score
def get_MNLP_score(result):
    # print(result["seq_score"])
    prob_seq = [ math.log1p(float(x)) if x!=0 else 0  for x in np.amax(result["probabilities"], axis=1)][:result["word_len"]]
    final_score = -np.sum(prob_seq)/result["word_len"]
    # final_score = -result["best_score"]/result["word_len"]
    # final_score = -result["best_score"]/len(prob_seq)
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

    result_counter=0
    best_probs=[]
    test_inpf = functools.partial(selected_input_fn, unlabelled_texts, unlabelled_tags,params)

    results = estimator.predict(test_inpf, yield_single_examples=True)
    # tf.logging.info("TOKENS",params["plan_token_count"])
    result_counter=0
    word_len_list=[]
    result_data=[]
    limit_addition = int(params["plan_sentence_count"] if use_sentence_base else params["plan_token_count"])
    # result_count=0
    for result in results:
        # print(result["log_probs"])
        # print("FLAT",result["log_probs"].flat)
        print("\nPREDICT UNLABELLED SAMPLES:",result_counter,'/',len(unlabelled_texts))


            # assert result_counter<len(silhouette_scores),"untrain_len:"+str(len(untrained_data))+'|'+str(result_counter)+'/'+str(len(silhouette_scores))+'|'+str(count)
        # if result_counter<len(silhouette_scores):
                # bald_context_score=bald*silhouette_scores[result_counter]
                # best_probs.append(bald_context_score)
        result["word_len"]=text_util.count_token([unlabelled_texts[result_counter]])
        word_len_list.append(result["word_len"])
        if use_LC:
            context_score=get_LC_score(result)
        elif use_MNLP:
            context_score=get_MNLP_score(result)
        best_probs.append(context_score)
        result_counter+=1
        prob_seq = [float(x) for x in np.amax(result["seq_score"], axis=1)]
        result_data.append({"tags":[x for x in result["tags"]],"seq_score":[float(x) for x in np.amax(result["seq_score"], axis=1)]})
    print(result_counter,len(unlabelled_texts))
    assert len(best_probs)>0, 'results must have been > 0'
            # print("SCORE",bald_context_score)
    best_probs = np.array(best_probs)
    file_util.dump(result_data,output_dir+"/prob_tags.pck")
    # top_k_idx = best_probs.argsort() [-top_k_num:][::-1]
    sorted_len_idx = np.argsort([x for x in word_len_list])[::-1]
    # best_probs = best_probs[sorted_len_idx]
    '''BEGIN TOKEN ANALYSYS WITHOUT SORT'''
    top_k_idx = best_probs.argsort()[-len(unlabelled_texts):][::-1][0:limit_addition]
    top_token_idx=[]
    token_sum=0
    # return sorted_len_idx[0:top_k_num]

    # for t_idx in top_k_idx:
    #     if not use_sentence_base:
    #         token_sum+=text_util.count_token([unlabelled_texts[t_idx]])
    #     else:
    #         token_sum+=1
    #     if token_sum<limit_addition:
    #         top_token_idx.append(t_idx)
    #     else:
    #         break
    '''END TOKEN ANALYSYS WITHOUT SORT'''
    #####
    '''BEGIN TOKEN ANALYSYS WITH SORT'''
    # current_token_sum=0
    # top_token_idx=[]
    # prob_sorted_idx = best_probs.argsort()[-len(best_probs):][::-1]
    # if not use_sentence_base:
    #     sort_score_by_len2(np.array(word_len_list), prob_sorted_idx, current_token_sum, params["plan_token_count"], top_token_idx, 0)
    # else:
    #     sort_score_by_len2(np.array(word_len_list), prob_sorted_idx, current_token_sum, params["plan_sentence_count"],
    #                        top_token_idx, 0)
    '''END TOKEN ANALYSYS WITH SORT'''

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

    #1 more step to convert tokens in 1 line to line


    with Path(output_dir + 'params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120,tf_random_seed=12)
    percent_list=[0.73,0.19, 0.77, 0.96, 1.92, 3.84, 3.84, 3.84, 3.84, 9.61, 9.6, 9.6, 9.6, 19.21, 19.2,5]
    percent_list=[1]*101
    # percent_list[0]=0.73
    # percent_list = [0.05, 1.64, 4.3, 7.63, 9.77, 12.23, 15.05, 17.65, 19.72, 24.69, 29.49, 34.84, 40.29, 54.65, 87.23,
    #                 100]
    # percent_list =[5]*21

    # percent_list = [x - percent_list[i - 1] if i > 0 else x for i, x in enumerate(percent_list)]
    al_sentence_amount_list=[101,141,299,454,748,1328,1859,2390,2933,4105]
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
    lines = read_data(os.path.join(dataset_dir,"train.txt"))
    all_texts_data = np.array([x[1] for x in lines])


    # all_formatted_training_data=all_texts_data
    all_tags_data=np.array([x[0] for x in lines])
    all_texts_data = np.array([x[1] for x in lines])

    # removal_texts = ([item for item, count in collections.Counter(all_texts_data).items() if count > 1])
    # removal_idx = [i for i, x in enumerate(all_texts_data) if x in removal_texts]
    # removal_id_dict = {x: [] for x in removal_texts}
    # for ri in removal_idx:
    #     removal_id_dict[all_texts_data[ri]].append(ri)
    # removal_tags = []
    # removal_texts1 = []
    # all_tags_data = np.array([x[0] for x in lines])
    # for key in removal_id_dict:
    #     removal_id_dict[key] = removal_id_dict[key][0]
    #     removal_tags.append(all_tags_data[removal_id_dict[key]])
    #     removal_texts1.append(key)
    # # all_formatted_training_data = all_texts_data
    #
    # all_texts_data = np.delete(all_texts_data, removal_idx)
    # all_tags_data = np.delete(all_tags_data, removal_idx)
    # all_texts_data = list(all_texts_data)
    # all_texts_data.extend(removal_texts1)
    # all_tags_data = list(all_tags_data)
    # all_tags_data.extend(removal_tags)
    # all_texts_data = np.array(all_texts_data)
    # all_tags_data = np.array(all_tags_data)
    #-----
    all_token_count = text_util.count_token(all_texts_data)
    all_data_len=len(all_texts_data)
    params["all_data_len"]=all_data_len
    for i,percent_value in enumerate(percent_list):
        params["loop_no"]=i


        loop_no=i
        params["all_token_count"] = all_token_count
        if i==0:
            # unlike_pct = batch_sample_size
            # seeds = generate_seed(all_texts_data, seed_pct)
            # seed_idx = [x["corpus_id"] for x in seeds]
            # seed_len=int(len(all_texts_data)*percent_list[0]/100)
            plan_token_count=all_token_count*percent_list[0]/100
            plan_sentence_count=int(math.ceil(len(all_texts_data)*percent_list[0]/100))
            seed_idx =file_util.load("/home/s4616573/data/bert_ner_output/data_simulation_al_gcn/"+data_name+".seeds."+str(seed_set_num)+".pck")
            # seed_tokens=0

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
        else:#COMBINE WITH BERT_RESULTS

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
            trained_texts = np.array(all_texts_data)[trained_indices]
            trained_tags = np.array(all_tags_data)[trained_indices]
            untrained_texts = [x for i_x, x in enumerate(all_texts_data) if i_x not in trained_indices]
            untrained_tags = [x for i_x, x in enumerate(all_tags_data) if i_x not in trained_indices]# trained_semtypes,namemention_dict=semtype_util.get_semtype_defs(trained_texts,use_i2b2)
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
            selected_idx = select_next_training_data(estimator,untrained_texts,untrained_tags, 0,params)

            assert len(selected_idx)>0,"Must be >0"
            selected_mapped_root_idx=match_idx_with_root_data(selected_idx,root_map_idx)
            file_util.dump(selected_mapped_root_idx,output_dir+"selected_idx_"+str(params["loop_no"])+".pck")
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






