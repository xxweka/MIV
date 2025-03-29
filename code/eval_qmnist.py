# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import sys
import ast
import cv2
import pickle
from absl import app, flags
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, classification_report

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

import tensorflow.keras as tk
import tensorflow.keras.layers as tkl
from modeling_cap import SplitFirst, Similarities, SimilaritiesDual, ConvEmbedLayer, \
    MaskedMHLayerNormalization, MaskedMHLinear, GlobalSumPoolingRagged1D, GlobalMaxPoolingRagged1D,\
    MilAttn, CrossAttnPooling, CrossDistAttnPooling, MSALayer, MaskedMultiply, HWChannelSE,\
    load_oneline, INPUT_KEYS, LABEL_KEYS, DELIMIT, TF_DTYPE, DTYPE, load_oneline_filenames

np.set_printoptions(suppress=True, threshold=sys.maxsize)
mycobj = {
    'tkl': tkl,
    'tk': tk,
    # 'gelu': activations.gelu,
    'MSALayer': MSALayer,
    'SplitFirst': SplitFirst,
    'Similarities': Similarities,
    'SimilaritiesDual': SimilaritiesDual,
    'ConvEmbedLayer': ConvEmbedLayer,
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
    'milpma': 'resid_attn',
}
IMAGE_SIZE = (32, 32)
MAX_NUM_TARGETS = 25


def load_images(input_dict, label_dict, shape_dict, stored_images):
    def load_img(image_files, get_first=False, row_id=None):
        fns = image_files.numpy().astype(str)
        # print(row_id.numpy(), get_first.numpy(), fns)
        loaded_images = []
        for image_file in fns:
            if image_file in stored_images:
                processed_img = stored_images[image_file]
            else:
                rawimg = cv2.imread(image_file, cv2.IMREAD_COLOR)
                h, w, _ = rawimg.shape
                if h != IMAGE_SIZE[1] or w != IMAGE_SIZE[0]:
                    processed_img = cv2.resize(rawimg, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
                else:
                    processed_img = rawimg
                stored_images[image_file] = processed_img
            loaded_images.append(processed_img)
        if get_first:
            assert len(loaded_images) == 1
            return loaded_images[0].astype(DTYPE)
        # print(row_id.numpy(), get_first.numpy(), tf.stack(loaded_images, axis=0).shape)
        return np.stack(loaded_images, axis=0).astype(DTYPE)

    query_img = tf.py_function(func=load_img, inp=[input_dict[INPUT_KEYS['query']], True], Tout=DTYPE)  #, input_dict['row_id']
    target_imgs = tf.py_function(func=load_img, inp=[input_dict[INPUT_KEYS['target']], False], Tout=DTYPE)  #, input_dict['row_id']
    query_img.set_shape(shape_dict[INPUT_KEYS['query']])
    target_imgs.set_shape(shape_dict[INPUT_KEYS['target']])

    inputs = {INPUT_KEYS['query']: query_img, INPUT_KEYS['target']: target_imgs}
    if 'cls' in input_dict:
        inputs['cls'] = input_dict['cls']
    return inputs, label_dict


def load_images_from_file(infile, same_digit, cls, load_fns, from_cache=None):
    with open(infile, 'r') as f:
        # num_examples = 0
        for line in f:
            # num_examples += 1
            if load_fns:
                inp, lbl = load_oneline_filenames(line, same_digit, cls)
            else:
                inp, lbl = load_oneline(line, IMAGE_SIZE, same_digit, cls, from_cache=from_cache)
            # inp['row_id'] = num_examples
            # if num_examples % 1000 == 0:
            #     print(num_examples, np.shape(inp[INPUT_KEYS['sig']]), np.shape(inp[INPUT_KEYS['anc']]), lbl, same_digit)
            yield inp, lbl


def load_images_from_list(inlist, same_digit, cls, load_fns, from_cache=None):
    for line in inlist:
        if load_fns:
            inp, lbl = load_oneline_filenames(line, same_digit, cls)
        else:
            inp, lbl = load_oneline(line, IMAGE_SIZE, same_digit, cls, from_cache=from_cache)
        yield inp, lbl


def load_pair_nocache(input_file, input_shape, batch_size=None, is_training=True, read_from='file',
                      same_digit=False, cls=False, shuffle_buffer=0, images_in_ram=None):
    assert read_from in ['file', 'list']
    # type_dict = {INPUT_KEYS['query']: TF_DTYPE, INPUT_KEYS['target']: TF_DTYPE}
    # otype = (type_dict, {LABEL_KEYS['main']: tf.int32})
    shape_dict = {INPUT_KEYS['query']: input_shape[INPUT_KEYS['query']],
                  INPUT_KEYS['target']: input_shape[INPUT_KEYS['target']]}
    if input_shape[INPUT_KEYS['target']][0] is not None:
        aux_shape = {INPUT_KEYS['query']: input_shape[INPUT_KEYS['query']],
                     INPUT_KEYS['target']: (None,)+input_shape[INPUT_KEYS['query']]}
    else:
        aux_shape = shape_dict
    pad_dict = {INPUT_KEYS['query']: 0., INPUT_KEYS['target']: 0.}
    type_dict = {INPUT_KEYS['query']: TF_DTYPE, INPUT_KEYS['target']: TF_DTYPE}
    type_inputs = {INPUT_KEYS['query']: tf.string, INPUT_KEYS['target']: tf.string}
    temp_shape = {INPUT_KEYS['query']: (1,), INPUT_KEYS['target']: (None,)}
    if cls:
        type_dict['cls'] = tf.int32
        type_inputs['cls'] = tf.int32
        shape_dict['cls'] = (1,)
        aux_shape['cls'] = (1,)
        temp_shape['cls'] = (1,)
        pad_dict['cls'] = 0
    # type_inputs['row_id'] = tf.int32
    # temp_shape['row_id'] = []
    oshape = (aux_shape, {LABEL_KEYS['main']: []})
    out_shape = (shape_dict, {LABEL_KEYS['main']: []})
    padding = (pad_dict, {LABEL_KEYS['main']: -1})
    interim_shape = (temp_shape, {LABEL_KEYS['main']: []})
    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # data_options.deterministic = True

    if images_in_ram is None:
        otype = (type_inputs, {LABEL_KEYS['main']: tf.int32})
        images_in_ram = {}
        if read_from == 'file':
            ds_meta = tf.data.Dataset.from_generator(
                load_images_from_file,
                output_types=otype,
                output_shapes=interim_shape,
                args=[input_file, same_digit, cls, True]
            )
        else:
            ds_meta = tf.data.Dataset.from_generator(
                lambda: load_images_from_list(input_file, same_digit, cls, True),
                output_types=otype,
                output_shapes=interim_shape
            )
        ds_meta = ds_meta.with_options(data_options).cache()  # OOM for large ds
        ds = ds_meta.map(lambda _x, _y: load_images(_x, _y, shape_dict, images_in_ram),
                         num_parallel_calls=tf.data.AUTOTUNE  # num_workers
                         )
    else:
        otype = (type_dict, {LABEL_KEYS['main']: tf.int32})
        if read_from == 'file':
            ds_meta = tf.data.Dataset.from_generator(
                # load_images_from_file,
                lambda: load_images_from_file(input_file, same_digit, cls, False, images_in_ram),
                output_types=otype,
                output_shapes=oshape,
                # args=[input_file, same_digit, cls, False, images_in_ram]
            )
        else:
            ds_meta = tf.data.Dataset.from_generator(
                lambda: load_images_from_list(input_file, same_digit, cls, False, images_in_ram),
                output_types=otype,
                output_shapes=oshape
            )
        ds = ds_meta.with_options(data_options)

    if is_training and shuffle_buffer != 0:
        reshuffle = (shuffle_buffer > 0)
        bufsz = abs(shuffle_buffer)
        ds = ds.shuffle(buffer_size=bufsz, reshuffle_each_iteration=reshuffle)
        print('load_pair_nocache: Shuffle/Reshuffle=', bufsz, reshuffle)
    elif is_training:
        print('load_pair_nocache: No shuffle')
    dataset = ds.padded_batch(
        batch_size,
        padding_values=padding,
        padded_shapes=out_shape
    )
    if is_training:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_images_cache(outfile, infile_list=None):
    def load_one_img(image_file):
        rawimg = cv2.imread(image_file, cv2.IMREAD_COLOR)
        h, w, _ = rawimg.shape
        if h != IMAGE_SIZE[1] or w != IMAGE_SIZE[0]:
            return cv2.resize(rawimg, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return rawimg

    if os.path.exists(outfile):
        with open(outfile, 'rb') as cf:
            images_cache = pickle.load(cf)
    else:
        images_cache = {}
        if infile_list is None:
            infile_list = []
        for infile in infile_list:
            assert infile.endswith('/')
            fns = os.listdir(infile)
            for fnm in fns:
                img_key = infile + fnm
                if img_key not in images_cache:
                    images_cache[img_key] = load_one_img(img_key)
        with open(outfile, 'wb') as cf:
            pickle.dump(images_cache, cf)
    return images_cache


def eval_classification(ds_test, model, num_classes, oh_encoder=None):
    # get test_ds labels
    y_val_enc = []
    # tf 2
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
    # print(keyinsts_idx, zeros_like)
    return zeros_like


def eval_explainability(eval_file, model_path, model, max_bag_seq=MAX_NUM_TARGETS):
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
        avg_mh = True
    else:
        layer_name = 'sim_pool'
        avg_mh = None
    num_exemplars = sum(1 for _ in open(eval_file))
    print(model_path, model_loaded['model_name'], ':', layer_name, ', Averaging multi-heads =', avg_mh, num_exemplars,
          flush=True)

    model_output = model.get_layer(layer_name).output
    idx_model = tk.Model(inputs=model.input, outputs=model_output)
    iauroc_list = []
    iap_list = []
    with open(eval_file, 'r') as ef:
        linecount = 0
        for line in ef:
            img_pair, labels = load_oneline(line, image_size=IMAGE_SIZE,
                                            cls=model_loaded['model_name'].startswith('milpma'))
            target_orig = img_pair[INPUT_KEYS['target']]
            bag_seq = len(target_orig)
            if max_bag_seq is not None:
                for d in range(len(target_orig.shape)):
                    if d == 0:
                        padwidth = ((0, max_bag_seq - bag_seq),)
                    else:
                        padwidth += ((0, 0),)
                target_vec = np.pad(img_pair[INPUT_KEYS['target']],
                                    pad_width=padwidth,
                                    mode='constant')
            else:
                target_vec = img_pair[INPUT_KEYS['target']]
            # create a batch from one line
            input_pair = {INPUT_KEYS['query']: np.expand_dims(img_pair[INPUT_KEYS['query']], axis=0),
                          INPUT_KEYS['target']: np.expand_dims(target_vec, axis=0)}
            if model_loaded['model_name'].startswith('milpma'):  # 'milmsa' not extracted
                input_pair['cls'] = np.expand_dims(img_pair['cls'], axis=0)
                # print(input_pair['cls'], flush=True)
            # predictions
            idx_output = idx_model.predict_on_batch(x=input_pair)
            # y = model.predict_on_batch(x=input_pair)
            inst_idx = idx_output[0] if output_atts else np.squeeze(idx_output, axis=-1)
            if model_loaded['model_name'].startswith('milpma'):
                inst_idx = inst_idx[:, -1]  # batch, q, head, k
            if max_bag_seq is not None:
                inst_idx = np.delete(inst_idx, np.s_[bag_seq:], axis=-1)
            # ground truth
            idx_gt = ast.literal_eval(line.split(DELIMIT)[-1])
            if idx_gt and (labels[LABEL_KEYS['main']] == 1):
                ki_pred = np.squeeze(inst_idx, axis=0)
                if len(ki_pred.shape) > 1 and output_atts and avg_mh:  # multi-head attn
                    ki_pred = np.mean(ki_pred, axis=0)
                if pure_baseline:
                    max_pred = np.zeros_like(ki_pred, dtype=ki_pred.dtype)
                    max_idx = np.argmax(ki_pred)
                    max_pred[max_idx] = 1.
                    ki_pred = max_pred.copy()
                # print(linecount, ':', line, idx_gt, inst_idx, ki_pred, flush=True)
                zrm = np.zeros_like(ki_pred, dtype=np.int32)
                kim = key_instances_mask(zrm, np.asarray(idx_gt)).flatten()
                kim_pred = ki_pred.flatten()
                # print(linecount, ':', zrm.shape, idx_gt)  #, kim)
                i_auroc = roc_auc_score(kim, kim_pred)
                i_ap = average_precision_score(kim, kim_pred)
                iauroc_list.append(i_auroc)
                iap_list.append(i_ap)
                # print(linecount, ':', 'iAUC =', i_auroc, ', iAP =', i_ap, ';', labels[LABEL_KEYS['main']], '\n', flush=True)
            linecount += 1
    assert len(iauroc_list) == len(iap_list)
    print('Average i-AUROC =', np.mean(iauroc_list), '; Average i-AP =', np.mean(iap_list), 'from sample-size =', len(iap_list))


flags.DEFINE_string('root_folder', 'qmnist/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('test', 'test_qmnist1.tsv', 'Test dataset under the root directory')
flags.DEFINE_string('model_file', 'ckpt/milmcase2heads-1se_resnet18tuning-excludeBN_035.hdf5',
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

    BATCH_SIZE = int(192 * 1.5)
    query_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)  # h, w, c
    target_seq = MAX_NUM_TARGETS
    inshape = {INPUT_KEYS['query']: query_shape, INPUT_KEYS['target']: (target_seq,) + query_shape}
    if model_name.startswith('milpma') or model_name.startswith('milmsa'):
        cls_placeholder = True
    else:
        cls_placeholder = False
    cache_filename = data_parent.split('/', 1)[0] + '.imgc'
    data_cache = build_images_cache(cache_filename, infile_list=['train60k/', 'test10k/'])
    ds_test = load_pair_nocache(test, input_shape=inshape, batch_size=BATCH_SIZE, is_training=False,
                                cls=cls_placeholder, images_in_ram=data_cache)

    eval_classification(ds_test, mymodel, num_classes=2)
    if eval_both:
        eval_explainability(test, model_path, mymodel, max_bag_seq=target_seq)


if __name__ == '__main__':
    app.run(main)
