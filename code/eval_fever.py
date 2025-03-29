# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import sys
import pickle
from absl import app, flags
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl
from transformers import AutoTokenizer

from modeling_cap import SplitFirst, Similarities, SimilaritiesDual, BERTEmbedLayer, MaskedMHLayerNormalization, \
    MaskedMHLinear, GlobalSumPoolingRagged1D, MilAttn, CrossAttnPooling, CrossDistAttnPooling, MSALayer, \
    MaskedMultiply, HWChannelSE, GlobalMaxPoolingRagged1D, load_oneline_tokens, INPUT_KEYS, LABEL_KEYS

np.set_printoptions(suppress=True, threshold=sys.maxsize)
mycobj = {
    'tkl': tkl,
    'tk': tk,
    # 'gelu': activations.gelu,
    'MSALayer': MSALayer,
    'SplitFirst': SplitFirst,
    'Similarities': Similarities,
    'SimilaritiesDual': SimilaritiesDual,
    'BERTEmbedLayer': BERTEmbedLayer,
    'GlobalSumPoolingRagged1D': GlobalSumPoolingRagged1D,
    'CrossAttnPooling': CrossAttnPooling,
    'CrossDistAttnPooling': CrossDistAttnPooling,
    'MaskedMHLayerNormalization': MaskedMHLayerNormalization,
    'MilAttn': MilAttn,
    'MaskedMHLinear': MaskedMHLinear,
    'HWChannelSE': HWChannelSE,
    'MaskedMultiply': MaskedMultiply,
    'GlobalMaxPoolingRagged1D': GlobalMaxPoolingRagged1D
}
pooler_name = {
    'milmcase': 'mca_pool',
    'milmcadse': 'mcad_pool',
    'milgattn': 'attn_scores',
    'milpma': 'resid_attn'
}
MAXIMUM_EVIDENCE = None  # 47  #
MAXIMUM_LEN = 96
sbert_pooler = 'cls'  # 'mean'  #


def npy_size(npyfile, num_classes):
    size=0
    freq = np.zeros([num_classes], dtype=np.float32)
    with open(npyfile, 'rb') as f:
        while True:
            try:
                datum = np.load(f, allow_pickle=True)
                size += 1
                freq[int(datum[2])] += 1.0
            except (EOFError, OSError, pickle.UnpicklingError):
                print('\nLoading completed')
                f.close()
                return size, freq/sum(freq)


def load_fever_npy(npyfile, cls, filter):
    with open(npyfile, 'rb') as f:
        while True:
            try:
                datum = np.load(f, allow_pickle=True)
                inputs, labels = load_oneline_tokens(
                    one_record=datum,
                    max_sentence_len=MAXIMUM_LEN,
                    max_num_evidences=MAXIMUM_EVIDENCE,
                    include_position=False,
                    cls=cls,
                    removals=filter
                )
                yield inputs, labels
            except (EOFError, OSError, pickle.UnpicklingError):
                print('\nLoading completed')
                f.close()
                return


def load_pair(input_file, input_shape, batch_size=None, is_training=True, shuffle_buffer=-1, cls=False, remove_list=None):
    type_dict = {INPUT_KEYS['query']: tf.int32, INPUT_KEYS['target']: tf.int32}
    shape_dict = {INPUT_KEYS['query']: input_shape[INPUT_KEYS['query']], INPUT_KEYS['target']: input_shape[INPUT_KEYS['target']]}
    pad_dict = {INPUT_KEYS['query']: 0, INPUT_KEYS['target']: 0}
    if cls:
        type_dict['cls'] = tf.int32
        shape_dict['cls'] = (1,)
        pad_dict['cls'] = 0
    otype = (type_dict, {LABEL_KEYS['main']: tf.int32})
    oshape = (shape_dict, {LABEL_KEYS['main']: []})
    padding = (pad_dict, {LABEL_KEYS['main']: -1})
    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # data_options.deterministic = True

    if remove_list is None:
        remove_list = []
    ds = tf.data.Dataset.from_generator(load_fever_npy,
                                        output_types=otype,
                                        output_shapes=oshape,
                                        args=[input_file, cls, remove_list]
                                        )
    ds = ds.with_options(data_options)
    if is_training:
        fn_split = input_file.split('.', 1)[0]
        fn_base = fn_split + '_cls' if cls else fn_split
        fn_ext = '.tf2c'
        fnm = fn_base + fn_ext
        ds = ds.cache(filename=fnm)
        if shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    dataset = ds.padded_batch(batch_size,
                              padding_values=padding,
                              padded_shapes=oshape
                              )
    if is_training:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def eval_classification(ds_test, model, num_classes, oh_encoder=None):
    # get test_ds labels
    y_val_enc = []
    for element in ds_test:
        _, y_val = element
        if isinstance(y_val, dict):
            y_batch = y_val[LABEL_KEYS['main']]
        else:
            y_batch = y_val
        y_val_enc.extend(y_batch)
    y_act = np.asarray(y_val_enc)
    if num_classes > 2:
        y_true = oh_encoder.transform(y_act.reshape(-1, 1))
        # find index of classes missing in val_ds (i.e. cols of oh-matrix whose elements are all 0)
        idx0 = np.where(~y_true.any(axis=0))[0]
        if len(idx0) > 0:
            y_act_oh = np.delete(y_true, idx0, axis=1)
        else:
            y_act_oh = y_true
    else:
        y_act_oh = y_act
    test_x = ds_test.map(lambda _x, _y: _x)
    pred = model.predict(test_x)  # np_array or a list of np_arrays
    if isinstance(pred, list):
        y_prob = pred[0]
    else:
        y_prob = pred
    print('\nTest data size =', y_prob.shape[0])
    assert y_act.shape[0] == y_prob.shape[0]
    if num_classes > 2:
        y_pred = tk.backend.eval(tf.argmax(y_prob, axis=1))  # axis=0 column-wise, =1 row-wise
    else:
        y_pred = tk.backend.eval(tf.argmax(np.append(1 - y_prob, y_prob, axis=-1), axis=1))  # axis 0=column-wise, 1=row-wise
    auroc = roc_auc_score(y_act_oh, y_prob)
    print('\nTest AUROC:', auroc)
    acc = accuracy_score(y_act, y_pred)
    print('Test Accuracy:', acc)
    print(classification_report(y_act, y_pred))
    print(confusion_matrix(y_act, y_pred))


def key_instances_mask(zeros_like, keyinsts_idx):
    if len(keyinsts_idx.shape) < len(zeros_like.shape):
        reshaped_idx = np.repeat(np.expand_dims(keyinsts_idx, 0), len(zeros_like), axis=0)
    else:
        reshaped_idx = keyinsts_idx
    np.put_along_axis(zeros_like, reshaped_idx, 1, axis=-1)
    return zeros_like


def eval_explainability(eval_file, model_path, model, max_bag_seq=None, max_word_seq=None, rmlist=None):
    mn = os.path.split(model_path)[1]
    assert not mn.startswith('milmsa')
    model_loaded = {'model_name': mn}
    for pn in pooler_name:
        if model_loaded['model_name'].startswith(pn):
            model_loaded['poolayer_name'] = pooler_name[pn]
    pure_baseline = mn.startswith('baseline') and (not mn.startswith('baseline_'))
    output_atts = False if pure_baseline else True

    # print out instance index
    if output_atts:
        layer_name = model_loaded['poolayer_name']
        avg_mh = True  # False  #
    else:
        layer_name = 'sim_pool'
        avg_mh = None
    model_output = model.get_layer(layer_name).output
    idx_model = tk.Model(inputs=model.input, outputs=model_output)
    print(model_path, model_loaded['model_name'], ':', layer_name, ', Averaging multi-heads =', avg_mh,
          flush=True)
    iauroc_list = []
    iap_list = []
    with open(eval_file, 'rb') as f:
        linecount = 0
        try:
            while True:
                datum = np.load(f, allow_pickle=True)
                input_raw, labels = load_oneline_tokens(
                    datum,
                    max_sentence_len=max_word_seq,
                    max_num_evidences=max_bag_seq,
                    include_position=True,
                    removals=rmlist,
                    cls=model_loaded['model_name'].startswith('milpma')
                )
                target_orig = input_raw[INPUT_KEYS['target']]
                bag_seq = len(target_orig)
                if max_word_seq is not None or max_bag_seq is not None:
                    pad0 = (0, max_bag_seq - bag_seq) if max_bag_seq is not None else (0, 0)
                    pad1 = (0, max_word_seq - len(target_orig[0])) if max_word_seq is not None else (0, 0)
                    target_vec = np.pad(target_orig, pad_width=(pad0, pad1), mode='constant')
                else:
                    target_vec = target_orig
                query_orig = input_raw[INPUT_KEYS['query']]
                if max_word_seq is not None:
                    pad0 = 0
                    pad1 = max_word_seq - len(query_orig)
                    query_vec = np.pad(query_orig, pad_width=(pad0, pad1), mode='constant')
                else:
                    query_vec = query_orig
                # create a batch from one line
                input_pair = {INPUT_KEYS['query']: np.expand_dims(query_vec, axis=0),
                              INPUT_KEYS['target']: np.expand_dims(target_vec, axis=0)}
                if model_loaded['model_name'].startswith('milpma'):
                    input_pair['cls'] = np.expand_dims(input_raw['cls'], axis=0)
                # predictions
                idx_output = idx_model.predict_on_batch(x=input_pair)
                inst_idx = idx_output[0] if output_atts else np.squeeze(idx_output, axis=-1)
                if model_loaded['model_name'].startswith('milpma'):
                    inst_idx = inst_idx[:, -1]
                if max_bag_seq is not None:
                    inst_idx = np.delete(inst_idx, np.s_[bag_seq:], axis=-1)

                # ground truth
                idx_gt = labels[LABEL_KEYS['scnd']]
                if idx_gt and (labels[LABEL_KEYS['main']] == 1):
                    if len(idx_gt) == bag_seq:
                        print(linecount, ':', 'All evidences are relevant, skipped', )
                    else:
                        ki_pred = np.squeeze(inst_idx, axis=0)
                        if len(ki_pred.shape) > 1 and output_atts and avg_mh:  # multi-head attn
                            ki_pred = np.mean(ki_pred, axis=0)
                            # print(ki_pred.shape, idx_gt)
                        # print(linecount, ':', inst_idx, ki_pred)
                        if pure_baseline:
                            max_pred = np.zeros_like(ki_pred, dtype=ki_pred.dtype)
                            max_idx = np.argmax(ki_pred)
                            max_pred[max_idx] = 1.
                            ki_pred = max_pred.copy()
                        zrm = np.zeros_like(ki_pred, dtype=np.int32)
                        igt = np.asarray(idx_gt)
                        kim = key_instances_mask(zrm, igt).flatten()
                        kim_pred = ki_pred.flatten()
                        # print(linecount, ':', kim, kim_pred, flush=True)
                        iauroc = roc_auc_score(kim, kim_pred)
                        iap = average_precision_score(kim, kim_pred)
                        iauroc_list.append(iauroc)
                        iap_list.append(iap)
                linecount += 1
        except (EOFError, pickle.UnpicklingError):
            print('\nLoading completed: ', linecount)
            f.close()
    assert len(iauroc_list) == len(iap_list)
    print('Average i-AUROC =', np.mean(iauroc_list), '; Average i-AP =', np.mean(iap_list), 'from sample-size =', len(iap_list))


flags.DEFINE_string('root_folder', 'FEVER/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('test', 'test_fever_all.npy', 'Test dataset under the root directory')
flags.DEFINE_string('model_file', 'ckpt/milmcase4heads-1se_SBERTmulti-qa-MiniLM-L6-cos-v1clspool_trainingNone_030.hdf5',
                    'Folder/path under root, and model(.h5) files')
flags.DEFINE_bool('eval_explainability', True, 'Whether to evaluate explainability quantitatively')
FLAGS = flags.FLAGS


def main(unused_argv):
    data_parent = FLAGS.root_folder
    os.chdir(data_parent)
    test = FLAGS.test
    model_path = FLAGS.model_file
    model_name = os.path.split(model_path)[1]
    eval_both = FLAGS.eval_explainability

    mymodel = tk.models.load_model(model_path, custom_objects=mycobj)
    line_shape = (None,)
    inshape = {INPUT_KEYS['query']: line_shape, INPUT_KEYS['target']: (MAXIMUM_EVIDENCE,) + line_shape}
    BATCH_SIZE = int(12 * 1.5)
    if model_name.startswith('milpma') or model_name.startswith('milmsa'):
        cls_placeholder = True
    else:
        cls_placeholder = False
    bert_pretrain_model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    bert_config_json = 'base_model/'
    sbert_tokenizer = AutoTokenizer.from_pretrained(bert_pretrain_model, cache_dir=bert_config_json)
    if sbert_pooler == 'mean':
        rmlist = sbert_tokenizer.all_special_ids
    else:
        rmlist = None
    ds_test = load_pair(test, input_shape=inshape, batch_size=BATCH_SIZE, is_training=False,
                        cls=cls_placeholder, remove_list=rmlist)

    eval_classification(ds_test, mymodel, num_classes=2)
    if eval_both:
        eval_explainability(
            test, model_path, mymodel,
            # max_bag_seq=MAXIMUM_EVIDENCE,
            # max_word_seq=MAXIMUM_LEN,
            rmlist=rmlist
        )


if __name__ == '__main__':
    app.run(main)
