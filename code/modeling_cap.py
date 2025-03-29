from __future__ import absolute_import, division, print_function, unicode_literals

import os
import subprocess as sp
import numpy as np
import cv2
import ast
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
import random
from datetime import datetime

from typing import Any, Optional
from absl import logging

import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl
# from tensorflow.python.keras.backend import set_session

# pip3 install tf-models-official
from official.nlp.modeling.layers import util, Transformer, PositionEmbedding, SelfAttentionMask
# from official.nlp.modeling.layers import TransformerEncoderBlock
from official.modeling import activations, tf_utils

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B3  # EfficientNetV2S  #
# pip install image-classifiers
from classification_models.tfkeras import Classifiers
# pip3 install transformers
# from transformers import TFBertModel, BertConfig
from transformers import TFAutoModel, AutoTokenizer

# tk.backend.set_image_data_format('channels_last')
DELIMIT = '\t'
DTYPE = np.dtype(np.float32)
TF_DTYPE = tf.dtypes.as_dtype(DTYPE)
INPUT_KEYS = {'query': 'query_input', 'target': 'target_input'}
LABEL_KEYS = {'main': 'binary_class', 'scnd': 'position'}
PRETRAIN_MODE = ['transfer-learning', 'tuning-excludeBN', 'tuning-fullBN', 'tuning-partBN', 're-training']


def create_virtual_gpus(vgpu_ram=None, onevgpu_per_physical=False, max_physicals=None, gpu_list=None):
    # tf.debugging.set_log_device_placement(True)
    # tf 1.x
    # conf = tf.compat.v1.ConfigProto(
    #     gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
    #     allow_soft_placement=True
    # )
    # sess = tf.compat.v1.Session(config=conf)
    # graph = tf.compat.v1.get_default_graph()
    # set_session(sess)

    vgpu_ram_buffer = 0.9
    # tf 2.x
    tf.config.set_soft_device_placement(True)
    command = "nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv"
    memory_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    content = [[n.split(',')[0].strip() for n in m.split()] for m in memory_info]
    avail_ram = {c[0]: float(c[1]) for c in content}
    free_ram = {c[0]: int(c[3]) for c in content}
    gpus = tf.config.list_physical_devices('GPU')
    usable_gpus = []
    for gpu in gpus:
        gpu_index = gpu.name.split(':')[-1].strip()
        if gpu_list is not None and gpu_index not in gpu_list:
            continue
        threshold = int(vgpu_ram / vgpu_ram_buffer) if vgpu_ram is not None else int(avail_ram[gpu_index] * 0.8)
        if free_ram[gpu_index] > threshold:
            if max_physicals is None or len(usable_gpus) < max_physicals:
                usable_gpus.append(gpu)
    try:
        tf.config.set_visible_devices(usable_gpus, 'GPU')
    except RuntimeError as rte:
        print(rte)
        usable_gpus = tf.config.get_visible_devices('GPU')
        pass
    print('GPU usability :', free_ram, '; GPUs with >', vgpu_ram, 'memory =', usable_gpus)

    if usable_gpus:
        for gpu in usable_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            if vgpu_ram is not None:
                nl = 1 if onevgpu_per_physical else free_ram[gpu.name.split(':')[-1].strip()] // vgpu_ram
                config_list = [tf.config.LogicalDeviceConfiguration(memory_limit=vgpu_ram)] * nl
                tf.config.set_logical_device_configuration(gpu, config_list)
        # try:
        # except (RuntimeError, ValueError) as err:
        #     print(err)
    return tf.config.list_logical_devices('GPU')


def _LARGE_NEG(tensor_type):
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1.e9


def safe_divide(numerator, denominator, epsilon=1e-6):
    dsign = tf.math.sign(denominator)
    # impute zeros
    imputed_sign = tf.math.add(tf.cast(tf.math.equal(dsign, 0.), dtype=dsign.dtype), dsign)
    indicator = tf.math.multiply(imputed_sign, tf.cast(tf.math.less(tf.math.abs(denominator), epsilon), dtype=denominator.dtype))
    safe_denom = tf.math.add(denominator, tf.math.multiply(indicator, epsilon))
    return tf.multiply(numerator, tf.math.reciprocal(safe_denom))


def load_oneline_tokens(one_record, max_sentence_len, max_num_evidences,
                        removals=None, include_position=False, cls=False):
    # if removals:
    if removals is not None and removals.size > 0:
        claim = [_w for _w in one_record[0] if _w not in removals]
        evids = [[_w for _w in evid if _w not in removals] for evid in one_record[1]]
    else:
        claim = one_record[0]
        evids = one_record[1]

    # unstruc: Pad words within a block, 2 equiv. ways
    # _y = np.asarray(list(itertools.zip_longest(*datum[1],fillvalue=0))).T
    max_len = max([len(d) for d in evids])
    if max_sentence_len is not None and max_len > max_sentence_len:
        max_len = max_sentence_len
    _y = tk.preprocessing.sequence.pad_sequences(evids,
                                                 maxlen=max_len,  # used to truncate
                                                 dtype='int32',
                                                 truncating='post',
                                                 padding='post')
    if max_num_evidences is not None:
        _y = _y[:max_num_evidences]

    _x = claim + [0,] * (max_len - len(claim))
    if len(_x) > max_len:
        _x = _x[:max_len]

    inputs = {INPUT_KEYS['query']: _x, INPUT_KEYS['target']: _y}
    if cls:
        inputs['cls'] = np.asarray([1], dtype=np.int32)

    _z = one_record[2]  # label
    if include_position:
        labels = {LABEL_KEYS['main']: _z, LABEL_KEYS['scnd']: one_record[3]}
    else:
        labels = {LABEL_KEYS['main']: _z}
    return inputs, labels


def load_oneline(line, image_size, masked_input=False, cls=False, from_cache=None):  # w,h
    def load_img(image_file):
        rawimg = cv2.imread(image_file, cv2.IMREAD_COLOR)
        h, w, _ = rawimg.shape
        if h != image_size[1] or w != image_size[0]:
            return cv2.resize(rawimg, image_size, interpolation=cv2.INTER_LANCZOS4)
        return rawimg

    entries = line.strip().split(DELIMIT)
    # print(masked_input, entries)
    if from_cache is not None:
        sign = from_cache[entries[0]]
    else:
        sign = load_img(entries[0])
    ancs = []
    templist = ast.literal_eval(entries[1])
    if masked_input:
        select = ast.literal_eval(entries[3])
        ancslist = [templist[s] for s in select]
    else:
        ancslist = templist
    for anc in ancslist:
        ancimg = from_cache[anc] if from_cache is not None else load_img(anc)
        ancs.append(ancimg)
    anch = np.stack(ancs, axis=0)
    inputs = {INPUT_KEYS['query']: np.asarray(sign, dtype=DTYPE),
              INPUT_KEYS['target']: np.asarray(anch, dtype=DTYPE)}
    if cls:
        inputs['cls'] = np.asarray([1,], dtype=np.int32)
    class_label = int(entries[2])
    labels = {LABEL_KEYS['main']: class_label}
    return inputs, labels


def load_oneline_filenames(line, masked_input=False, cls=False):
    entries = line.strip().split(DELIMIT)
    # print(masked_input, entries)
    query = entries[0]
    templist = ast.literal_eval(entries[1])
    if masked_input:
        select = ast.literal_eval(entries[3])
        target = [templist[s] for s in select]
    else:
        target = templist
    inputs = {INPUT_KEYS['query']: [query,],
              INPUT_KEYS['target']: target}
    if cls:
        inputs['cls'] = np.asarray([1,], dtype=np.int32)
    class_label = int(entries[2])
    labels = {LABEL_KEYS['main']: class_label}
    return inputs, labels


def _masked_variance(values, mask_identity, axis, keep_dims=False):
    _, variance = _masked_moments(values, mask_identity=mask_identity, axis=axis, keep_dims=True)
    if not keep_dims:
        return_var = tf.squeeze(variance, axis=axis)
    else:
        return_var = variance
    return return_var


# return (0., 0.) if masked
def _masked_moments(values, mask_identity, axis, keep_dims=False):
    assert len(mask_identity.shape) == len(values.shape)
    inv_sumofweighting = tf.math.reciprocal_no_nan(tf.math.reduce_sum(mask_identity, axis=axis, keepdims=True))
    masked_values = tf.math.multiply(values, mask_identity)
    # tf.debugging.assert_all_finite(masked_values, message='masked_moments: masked values check finite')
    mean = tf.math.reduce_sum(tf.math.multiply(masked_values, inv_sumofweighting), axis=axis, keepdims=True)
    # tf.debugging.assert_all_finite(mean, message='masked_moments: mean check finite')
    masked_mean = tf.math.multiply(mean, mask_identity)
    ssq = tf.math.multiply(tf.math.square(tf.math.subtract(masked_values, masked_mean)), mask_identity)
    # tf.debugging.assert_all_finite(ssq, message='masked_moments: ssq check finite')
    variance = tf.math.reduce_sum(tf.math.multiply(ssq, inv_sumofweighting), axis=axis, keepdims=True)
    # tf.debugging.assert_all_finite(variance, message='masked_moments: variance check finite')

    if not keep_dims:
        return tf.squeeze(mean, axis=axis), tf.squeeze(variance, axis=axis)
    return mean, variance


def _split_heads(x, split_list):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    newshape = tf.concat([tf.shape(x)[:-1], split_list], axis=-1)
    return tf.reshape(x, newshape)


class CkptUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_fullpath):
        super().__init__()
        self.ckpt_fullpath = ckpt_fullpath

    def on_epoch_begin(self, epoch, logs=None):
        ckpt_path = eval("f'{}'".format(self.ckpt_fullpath))
        ckpt_folder, ckpt_file = os.path.split(ckpt_path)
        if os.path.exists(ckpt_path):
            with open(ckpt_folder + '/checkpoint', 'w') as cf:
                cf.write('model_checkpoint_path: "' + ckpt_file + '"\n' +
                         'all_model_checkpoint_paths: "' + ckpt_file + '"')


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds, y_actual, num_cls: int, pred_idx: int, start_time, num_batches_per_epoch,
                 topn_prob=5, split=False):  # assume class labels are encoded
        super().__init__()
        # print('val:', num_cls, val_lbltg, pred_idx)
        self.label_idx = pred_idx
        self.num_classes = num_cls
        self.num_batches_per_epoch = num_batches_per_epoch
        self.topn_prob = topn_prob
        # save x values
        self.val_ds = ds
        if split:
            idx = list(range(len(y_actual)))
            random.shuffle(idx)
            self.split = [idx[:len(idx) // 2], idx[len(idx) // 2:]]
        else:
            self.split = None
        self.y_act = y_actual
        self.idx0 = None
        if self.num_classes > 2:
            ohe = preprocessing.OneHotEncoder(categories=[np.arange(self.num_classes)], sparse=False,
                                              handle_unknown='ignore')
            y_true = ohe.fit_transform(self.y_act.reshape(-1, 1))
            self.idx0 = np.where(~y_true.any(axis=0))[
                0]  # find index of classes missing in val_ds (i.e. cols of oh-matrix whose elements are all 0)
            if len(self.idx0) > 0:
                self.y_act_oh = np.delete(y_true, self.idx0, axis=1)
            else:
                self.y_act_oh = y_true
        else:
            self.y_act_oh = self.y_act  # for 2-class, oh is same as class label (0, 1)
        if split:
            self.val_test_act = [self.y_act[self.split[0]], self.y_act[self.split[1]]]
            self.val_test_actoh = [self.y_act_oh[self.split[0]], self.y_act_oh[self.split[1]]]
        self.start_time = start_time

    def on_epoch_begin(self, epoch, logs=None):
        if tf.version.VERSION.startswith('2.11'):
            lr = self.model.optimizer.learning_rate.numpy()
            assert lr.dtype is DTYPE
            print('\nCurrent lr at epoch_id', epoch, '=', lr)
        else:
            lr = self.model.optimizer.lr
            if isinstance(lr, tk.optimizers.schedules.LearningRateSchedule):
                print('\nCurrent lr at epoch_id', epoch, '=', tk.backend.eval(lr(epoch * self.num_batches_per_epoch)))
            else:
                print('\nCurrent lr at epoch_id', epoch, '=', tk.backend.get_value(lr))

    def on_epoch_end(self, epoch, logs=None):
        # print('before predict', tk.backend.eval(tk.backend.learning_phase())) #False
        pred = self.model.predict(self.val_ds)  # np_array or a list of np_arrays
        # prob of "1" (ncls=2) or vector of prob (ncls>2) of whole ds
        if isinstance(pred, list):
            y_prob = pred[self.label_idx]
        else:
            y_prob = pred

        # print('after predict', tk.backend.eval(tk.backend.learning_phase())) #False
        print('\nTest data size =', y_prob.shape[0])
        assert self.y_act.shape[0] == y_prob.shape[0]
        printout = ''
        if self.num_classes > 2:
            maxprob = np.amax(y_prob, axis=0)  # axis=0 column-wise, =1 row-wise
            for c in range(y_prob.shape[1]):
                printout += 'max(prob[y=' + str(c) + '])=' + str(maxprob[c]) + '; '
            y_pred = tk.backend.eval(tf.argmax(y_prob, axis=1))  # axis=0 column-wise, =1 row-wise
            if len(self.idx0) > 0:
                y_prob = np.delete(y_prob, self.idx0, axis=1)  # axis=0 del row, =1 del column
        else:
            for c in range(self.num_classes):
                p = np.squeeze(y_prob[self.y_act == c])
                if self.topn_prob > len(p):
                    topn = len(p)
                else:
                    topn = self.topn_prob
                printout += \
                    'Top ' + str(topn) + \
                    ' max(prob[y=' + str(c) + '])=' + str(p[np.argpartition(p, -topn)[-topn:]]) + '; ' + \
                    ' min(prob[y=' + str(c) + '])=' + str(p[np.argpartition(p, topn)[:topn]])
                    # np.amax()
                # printout += 'max(prob[y=' + str(c) + '])=' + str(np.amax(y_prob[self.y_act == c])) + '; '
            y_pred = tk.backend.eval(
                tf.argmax(np.append(1 - y_prob, y_prob, axis=-1), axis=1))  # axis=0 column-wise, =1 row-wise
        print(printout.strip()[:-1])  # print interim prob info
        # print('\n', self.y_act.shape, self.y_act_oh.shape, y_pred.shape)

        if self.split is not None:
            # validate & test
            val_test_prob = [y_prob[self.split[0]], y_prob[self.split[1]]]
            val_test_pred = [y_pred[self.split[0]], y_pred[self.split[1]]]
            for i, st in enumerate(['Val Data', 'Test Data']):
                print(st + ':')
                auroc = roc_auc_score(self.val_test_actoh[i], val_test_prob[i])
                print('\nTest AUROC:', auroc)
                acc = accuracy_score(self.val_test_act[i], val_test_pred[i])
                print('Test Accuracy:', acc)
                print(classification_report(self.val_test_act[i], val_test_pred[i]))
                print(confusion_matrix(self.val_test_act[i], val_test_pred[i]))
        else:
            # combined
            auroc = roc_auc_score(self.y_act_oh, y_prob)
            print('\nTest AUROC:', auroc)
            acc = accuracy_score(self.y_act, y_pred)
            print('Test Accuracy:', acc)
            print(classification_report(self.y_act, y_pred))
            print(confusion_matrix(self.y_act, y_pred))
        print('Time taken so far =', datetime.now() - self.start_time)


class ConvEmbedLayer(tkl.Layer):
    def __init__(self, image_shape, pretrain, pooling=None, model_name=None, **kwargs):
        super(ConvEmbedLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.temp_masking = tkl.Masking(mask_value=0., name='conv_embed_tempmasking')
        assert pretrain in PRETRAIN_MODE
        pretrain_params = {
            # weights, trainable, excludeBN, self.training_mode
            PRETRAIN_MODE[0]: ('imagenet', False, False, False),
            PRETRAIN_MODE[1]: ('imagenet', True, True, None),
            PRETRAIN_MODE[2]: ('imagenet', True, False, None),
            PRETRAIN_MODE[3]: ('imagenet', True, False, False),
            PRETRAIN_MODE[4]: (None, True, False, None)
        }
        self.pooling = pooling
        pretrained_weights, conv_trainable, excludeBN, self.training_mode = pretrain_params[pretrain]
        self.model_name = model_name
        if model_name.startswith('EfficientNetV2'):
            # self.base_model = EfficientNetV2S(
            self.base_model = EfficientNetV2B3(
                weights=pretrained_weights,
                input_shape=image_shape,
                include_top=False,
                pooling=pooling,
                include_preprocessing=True
            )
            self.preprocess_input = preprocess_input  # pass-through
            self.pool_layer = None
        elif model_name.startswith('resnet'):
            vmodel, self.preprocess_input = Classifiers.get(model_name)
            self.base_model = vmodel(
                include_top=False,
                weights=pretrained_weights,
                input_shape=image_shape
            )
            if pooling == 'avg':
                self.pool_layer = tkl.GlobalAveragePooling2D()
            elif pooling is None:
                self.pool_layer = None
            else:
                raise ValueError('Invalid pooling method!')
        else:
            raise ValueError('Incorrect base_model name!')
        self.base_model.trainable = conv_trainable
        if self.base_model.trainable and excludeBN:
            for lyr in self.base_model.layers:
                if isinstance(lyr, tkl.BatchNormalization):
                    lyr.trainable = False  # TF2: BN.trainable=F automatically sets BN.training=F
                else:
                    lyr.trainable = True
        shape_dim = -3 if self.pooling is None else -1
        self.lastdims = self.base_model.get_layer(index=-1).output_shape[shape_dim:]
        # print(self.lastdims, self.pooling)
        self._config_dict = {
            'image_shape': image_shape,
            'pretrain': pretrain,
            'model_name': model_name,
            'pooling': pooling
        }

    def compute_output_shape(self, input_shape):
        return input_shape[:-3] + self.lastdims

    def compute_mask(self, inputs, mask=None):
        # assert mask is not None
        if mask is None:
            mask = self._make_mask(inputs)
        ret_mask = tf.math.reduce_any(mask, axis=(-2, -1), keepdims=False)
        if self.pooling is None:
            for _ in range(len(self.lastdims[:-1])):
                ret_mask = tf.expand_dims(ret_mask, axis=-1)
            mask_outshape = tf.concat([tf.shape(inputs)[:-3], self.lastdims[:-1]], axis=-1)
            ret_mask = tf.broadcast_to(ret_mask, mask_outshape)
        return ret_mask

    def call(self, inputs, mask=None, training=None):
        # assert mask is not None
        if mask is None:
            mask = self._make_mask(inputs)
        reduced_mask = tf.math.reduce_any(mask, axis=(-2, -1), keepdims=False)
        masked_inputs = tf.boolean_mask(inputs, mask=reduced_mask) # flattened OK, batch*seq
        indices = tf.cast(tf.where(reduced_mask), tf.int32)

        if self.model_name.startswith('resnet'):
            y = self.preprocess_input(masked_inputs)
        else:
            y = self.preprocess_input(masked_inputs, data_format='channels_last')
        training_mode = training if self.training_mode is None else False
        # print('Base_model training mode =', training_mode)
        y = self.base_model(y, training=training_mode)
        if self.pool_layer is not None:
            y = self.pool_layer(y)
        # tf.debugging.assert_all_finite(y, message='ConvEmbed: base_model output check finite')
        # clipped_y = tf.clip_by_value(y, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)

        raw_inshape = tf.shape(inputs)
        outshape = tf.concat([raw_inshape[:-3], self.lastdims], axis=-1)
        y_output = tf.scatter_nd(indices, y, shape=outshape)
        # tf.debugging.assert_all_finite(y_output, message='ConvEmbed: output check finite')
        return y_output

    def get_config(self):
        config = super(ConvEmbedLayer, self).get_config()
        config.update(self._config_dict)
        return config

    def _make_mask(self, input):
        return self.temp_masking.compute_mask(input)


class BERTEmbedLayer(tkl.Layer):
    def __init__(self, config_file, pretrained_file, bert_trainable, pooling=None, training_mode=None, **kwargs):
        super(BERTEmbedLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.masking = tkl.Masking(mask_value=0, name='bert_embed_masking')
        # bert_config = BertConfig.from_json_file(config_file)
        # self.base_model = TFBertModel.from_pretrained(pretrained_file, config=bert_config, from_pt=True)
        self.base_model = TFAutoModel.from_pretrained(pretrained_file, cache_dir=config_file)
        self.lastdims = (self.base_model.config.hidden_size,)
        self.base_model.trainable = bert_trainable
        if pooling is None:
            self.pooler = 'cls'
        else:
            self.pooler = pooling
        self.training_mode = training_mode
        # print(self.lastdims)
        self._config_dict = {
            'config_file': config_file,
            'pretrained_file': pretrained_file,
            'bert_trainable': bert_trainable,
            'training_mode': training_mode,
            'pooling': pooling
        }

    def _make_mask(self, inputs):
        return self.masking.compute_mask(tf.expand_dims(inputs, axis=-1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + self.lastdims

    def compute_mask(self, inputs, mask=None):
        return tf.math.reduce_any(self._make_mask(inputs), axis=-1)

    def call(self, inputs, mask=None, training=None):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        _mask = self._make_mask(inputs)
        reduced_mask = tf.math.reduce_any(_mask, axis=-1)
        # print(reduced_mask, _mask, inputs)
        masked_inputs = tf.boolean_mask(inputs, mask=reduced_mask)  # flattened OK, batch*seq
        masked_mask = tf.boolean_mask(_mask, mask=reduced_mask)
        # print(masked_inputs, masked_mask)
        indices = tf.cast(tf.where(reduced_mask), tf.int32)
        attn_mask = tf.cast(masked_mask, dtype=masked_inputs.dtype)

        training_mode = training if self.training_mode is None else False
        # print('Base_model training mode =', training_mode, ', SBert-pooling =', self.pooler)
        sbert_output = self.base_model(
            input_ids=masked_inputs,
            attention_mask=attn_mask,
            token_type_ids=None,
            position_ids=None,
            training=training_mode,
            output_attentions=False
        )
        if self.pooler == 'cls':
            y = sbert_output[0][:, 0]
        elif self.pooler == 'mean':
            y = self.mean_pooling(sbert_output[0], attn_mask)
        else:
            raise NotImplementedError(f'Invalid pooling for SBERT: {self.pooler}')
        # # y = bert_output.pooler_output

        raw_inshape = tf.shape(inputs)
        outshape = tf.concat([raw_inshape[:-1], self.lastdims], axis=-1)
        y_output = tf.scatter_nd(indices, y, shape=outshape)
        return y_output

    def get_config(self):
        config = super(BERTEmbedLayer, self).get_config()
        config.update(self._config_dict)
        return config

    def mean_pooling(self, outputs, mask):
        mask_weights = tf.expand_dims(tf.cast(mask, dtype=outputs.dtype), axis=-1) + tf.zeros_like(outputs,
                                                                                                   dtype=outputs.dtype)
        mask_weights.set_shape(outputs.shape)
        sum_inputs = tf.math.reduce_sum(tf.math.multiply(outputs, mask_weights), axis=-2)
        sum_weights = tf.math.reduce_sum(mask_weights, axis=-2)
        return tf.math.multiply(sum_inputs, tf.math.reciprocal(sum_weights))


class PostConvFlatten(tkl.Layer):
    def __init__(self, output_units=None, output_actv=None, l2norm=False, **kwargs):
        super(PostConvFlatten, self).__init__(**kwargs)
        self.supports_masking = True
        self.output_units = output_units
        self.output_actv = output_actv
        self.l2norm = l2norm
        self._config_dict = {
            'output_units': output_units,
            'output_actv': output_actv,
            'l2norm': l2norm
        }
        self.flatten = tkl.Flatten()

    def build(self, input_shape):
        if self.output_units is not None and self.output_units > 0:
            fcn_trainable = True
            units = self.compute_output_shape(input_shape)[-1]
            self.fcn = tkl.Dense(units=units, activation=self.output_actv, trainable=fcn_trainable)
        else:
            self.fcn = None
        super(PostConvFlatten, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.output_units is not None and self.output_units > 0:
            return input_shape[:-3] + (self.output_units,)
        return input_shape[:-3] + self.flatten.compute_output_shape(input_shape)[-1:]

    def compute_mask(self, inputs, mask=None):
        assert mask is not None
        return tf.math.reduce_any(mask, axis=(-1,-2))

    def call(self, inputs, mask=None):
        assert mask is not None
        raw_inshape = tf.shape(inputs)
        reduced_mask = tf.math.reduce_any(mask, axis=(-1,-2))
        masked_inputs = tf.boolean_mask(inputs, mask=reduced_mask) # flattened OK, batch*seq
        indices = tf.cast(tf.where(reduced_mask), tf.int32)

        y = self.flatten(masked_inputs)
        if self.output_units is not None and self.output_units > 0:
            outshape = tf.concat([raw_inshape[:-3], tf.expand_dims(self.output_units, axis=-1)], axis=-1)
            y = self.fcn(y)
        else:
            outshape = tf.concat([raw_inshape[:-3], y.shape[-1:]], axis=-1)
        if self.l2norm:
            y = tf.math.l2_normalize(y, axis=-1)

        y_output = tf.scatter_nd(indices, y, shape=outshape)
        return y_output

    def get_config(self):
        config = super(PostConvFlatten, self).get_config()
        config.update(self._config_dict)
        return config


class MaskedMHLayerNormalization(tkl.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-6,
                 center=True,
                 scale=True,
                 num_heads=1,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MaskedMHLayerNormalization, self).__init__(name=name, trainable=trainable, **kwargs)
        # axis is normalization dims, while gamma/beta is always on channel/last dim
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise ValueError('Expected an int or a list/tuple of ints for the '
                             'argument \'axis\', but received instead: %s' % axis)

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.num_heads = num_heads
        self.beta_initializer = tk.initializers.get(beta_initializer)
        self.gamma_initializer = tk.initializers.get(gamma_initializer)
        self.beta_regularizer = tk.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tk.regularizers.get(gamma_regularizer)
        self.beta_constraint = tk.constraints.get(beta_constraint)
        self.gamma_constraint = tk.constraints.get(gamma_constraint)
        self.supports_masking = True
        self._config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'num_heads': self.num_heads,
            'beta_initializer': tk.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tk.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tk.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tk.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tk.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tk.constraints.serialize(self.gamma_constraint)
        }

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def build(self, input_shape):
        inp_shape, stats_shape = input_shape
        ndims = len(stats_shape)
        if ndims is None:
            raise ValueError('Input shape %s has undefined rank.' % stats_shape)
        assert ndims == len(inp_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x
        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

        param_shape = (inp_shape[-1],)  # always on channel dim
        if self.scale:
            self.gamma = self.add_weight(
                name='ln_gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                # dtype=TF_DTYPE,
                # experimental_autocast=False,
                trainable=True
            )
        else:
            self.gamma = tf.ones(shape=param_shape, dtype=TF_DTYPE)
        if self.center:
            self.beta = self.add_weight(
                name='ln_beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                # dtype=TF_DTYPE,
                # experimental_autocast=False,
                trainable=True
            )
        else:
            self.beta = tf.zeros(shape=param_shape, dtype=TF_DTYPE)
        self.built = True

    def call(self, inputs, mask=None):
        def create_mask_weights(raw_inputs, raw_mask):
            indtype = raw_inputs.dtype
            shape_placeholder = tf.zeros_like(raw_inputs, dtype=indtype)
            if raw_mask is not None:
                mask_weights = tf.expand_dims(tf.cast(raw_mask, dtype=indtype), axis=-1) + shape_placeholder
            else:
                mask_weights = tf.ones_like(shape_placeholder, dtype=indtype)
            return tf.ensure_shape(mask_weights, shape_placeholder.shape)

        inp, stats = inputs
        if mask is not None:
            mask_inp, mask_stats = mask
        else:
            mask_inp, mask_stats = None, None

        mask_weights_stats = create_mask_weights(stats, mask_stats)
        # multi-head splitting
        split_stats = [self.num_heads, mask_weights_stats.shape[-1] // self.num_heads]
        res_stats = tf.transpose(_split_heads(stats, split_list=split_stats), perm=[0, 2, 1, 3])
        res_mask_stats = tf.transpose(_split_heads(mask_weights_stats, split_list=split_stats), perm=[0, 2, 1, 3])
        res_axis = []
        for d in self.axis:
            if d > 0:
                res_axis.append(d+1)
            else:
                res_axis.append(d)
        mean, variance = _masked_moments(res_stats, axis=res_axis, mask_identity=res_mask_stats, keep_dims=True)
        # mean, variance = tf.nn.weighted_moments(inputs, axes=mh_axis, frequency_weights=mask_weights, keepdims=True)
        # tf.debugging.assert_all_finite(mean, message='LN: mean check finite')
        # tf.debugging.assert_non_negative(variance, message='LN: var check >=0')
        std = tf.math.sqrt(variance)

        mask_weights_inp = create_mask_weights(inp, mask_inp)
        split_inp = [self.num_heads, mask_weights_inp.shape[-1] // self.num_heads]
        res_inputs = tf.transpose(_split_heads(inp, split_list=split_inp), perm=[0, 2, 1, 3])
        output_shape = tf.shape(inp)

        normed = safe_divide(numerator=tf.math.subtract(res_inputs, mean), denominator=std, epsilon=self.epsilon)
        res_normed = tf.reshape(tf.transpose(normed, perm=[0, 2, 1, 3]), shape=output_shape)
        # tf.debugging.assert_all_finite(std, message='LN: std check finite')
        # tf.debugging.assert_all_finite(normed, message='LN: safe_div check finite')

        tf.debugging.assert_all_finite(self.gamma, message='LN: gamma check finite')
        tf.debugging.assert_all_finite(self.beta, message='LN: beta check finite')
        output = tf.math.add(tf.math.multiply(res_normed, self.gamma), self.beta)
        # clipped_output = tf.clip_by_value(output, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)
        return tf.math.multiply(output, mask_weights_inp)

    def get_config(self):
        config = super(MaskedMHLayerNormalization, self).get_config()
        config.update(self._config)
        return config


class MaskedMHLinear(tkl.Layer):
    def __init__(self, unit, num_heads=1, use_bias=True, kernel_initializer=None, activate=False,
                 post_norm=False, norm_axis=None, norm_adjust=False, epsilon=None, **kwargs):
        super(MaskedMHLinear, self).__init__(**kwargs)
        self.supports_masking = True
        self.unit = unit
        self.num_heads = num_heads
        assert not post_norm or (norm_axis is not None and epsilon is not None)
        self.post_norm = post_norm
        self.epsilon = epsilon
        self.norm_adjust = norm_adjust
        self.norm_axis = norm_axis
        self.use_bias = use_bias
        self.kernel_init = kernel_initializer
        self.activate = activate
        self._config_dict = {
            'unit': unit,
            'use_bias': use_bias,
            'kernel_initializer': kernel_initializer,
            'activate': activate,
            'num_heads': num_heads,
            'post_norm': post_norm,
            'norm_axis': norm_axis,
            'norm_adjust': norm_adjust,
            'epsilon': epsilon
        }

    def build(self, input_shape):
        vec_size = input_shape[-1]
        assert (self.unit % self.num_heads == 0) and (vec_size % self.num_heads == 0)
        params_depth = self.unit // self.num_heads
        param_shape = (self.num_heads, params_depth)
        if self.kernel_init is None:
            initi = np.reshape(np.eye(vec_size, self.unit, dtype=DTYPE), (vec_size,)+param_shape)
            self.kernel_init = tk.initializers.Constant(initi)
        self.kernel = self.add_weight(name='linear_weights',
                                      shape=(vec_size,)+param_shape,
                                      initializer=self.kernel_init,
                                      trainable=True
                                      )
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=param_shape,
                                        initializer='zeros',
                                        trainable=True
                                        )
        else:
            self.bias = tf.zeros(shape=param_shape, dtype=TF_DTYPE)
        if self.post_norm and self.norm_adjust:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer='ones',
                trainable=True
            )
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer='zeros',
                trainable=True
            )
        else:
            self.gamma = None
            self.beta = None
        super(MaskedMHLinear, self).build(input_shape)

    # def compute_output_shape(self, input_shape):
    #     return input_shape[:-1] + (self.unit,)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        indtype = inputs.dtype
        _xw = tf.einsum('...c,chd->...hd', inputs, self.kernel)  # (..., h, unit // num_heads)
        if self.activate:
            head_linear = tf.math.tanh(tf.math.add(_xw, self.bias))
            # head_linear = tf.math.add(_split_heads(inputs, list(self.bias.shape)), tf.math.add(_xw, self.bias))
        else:
            head_linear = tf.math.add(_xw, self.bias)

        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.unit,]], axis=-1)
        if mask is not None:
            mask_weights = tf.expand_dims(tf.cast(mask, dtype=indtype), axis=-1) + tf.zeros(shape=output_shape, dtype=indtype)
        else:
            mask_weights = tf.ones(shape=output_shape, dtype=indtype)
        mask_weights.set_shape(inputs.shape[:-1] + (self.unit,))

        if self.post_norm:
            # multi-head splitting
            split = list(self.bias.shape)
            res_mask = tf.transpose(_split_heads(mask_weights, split_list=split), perm=[0, 2, 1, 3])
            res_inputs = tf.transpose(head_linear, perm=[0, 2, 1, 3])
            mean, variance = _masked_moments(res_inputs, axis=self.norm_axis, mask_identity=res_mask, keep_dims=True)
            std = tf.math.sqrt(variance)
            normed = safe_divide(numerator=tf.math.subtract(res_inputs, mean), denominator=std, epsilon=self.epsilon)
            if self.gamma is not None and self.beta is not None:
                res2b = tf.math.add(tf.math.multiply(tf.transpose(normed, perm=[0, 2, 1, 3]), self.gamma), self.beta)
            else:
                res2b = tf.transpose(normed, perm=[0, 2, 1, 3])
        else:
            res2b = head_linear

        output = tf.reshape(res2b, shape=output_shape)
        return tf.math.multiply(mask_weights, output)

    def get_config(self):
        config = super(MaskedMHLinear, self).get_config()
        config.update(self._config_dict)
        return config


class GlobalSumPoolingRagged1D(tkl.Layer):
    def __init__(self, keepdims=False, **kwargs):
        super(GlobalSumPoolingRagged1D, self).__init__(**kwargs)
        self.supports_masking = True
        self.keepdims = keepdims
        self._config_dict = {
            'keepdims': keepdims
        }

    def compute_mask(self, inputs, mask=None):
        assert mask is not None
        return tf.math.reduce_any(mask, axis=-1, keepdims=self.keepdims)

    def call(self, inputs, mask=None, training=None):  # inputs = [sig, ancs]
        assert mask is not None
        mask_weights = tf.expand_dims(tf.cast(mask, dtype=inputs.dtype), axis=-1) + tf.zeros_like(inputs, dtype=inputs.dtype)
        mask_weights.set_shape(inputs.shape)

        reduced_sum = tf.math.reduce_sum(tf.math.multiply(inputs, mask_weights), axis=-2, keepdims=self.keepdims)
        # tf.debugging.assert_all_finite(reduced_sum, message='sum-pool: check output nan')
        # return tf.clip_by_value(reduced_sum, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)
        return reduced_sum

    def get_config(self):
        config = super(GlobalSumPoolingRagged1D, self).get_config()
        config.update(self._config_dict)
        return config


class GlobalMaxPoolingRagged1D(tkl.Layer):
    def __init__(self, **kwargs):
        super(GlobalMaxPoolingRagged1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        assert mask is not None
        return tf.math.reduce_any(mask, axis=-1)

    def call(self, inputs, mask=None, training=None): # inputs = [sig, ancs]
        assert mask is not None
        mask_weights = tf.expand_dims(tf.cast(mask, dtype=inputs.dtype), axis=-1) + tf.zeros_like(inputs)
        mask_weights.set_shape(inputs.shape)
        _input_min = tf.math.minimum(tf.math.reduce_min(inputs), 0.)
        _neg_inf = _input_min + _LARGE_NEG(inputs.dtype)
        masked_inputs = mask_weights * inputs + (1.-mask_weights) * _neg_inf
        # tf.debugging.assert_all_finite(inputs, message='max-pool: check nan')

        reduced_max = tf.math.reduce_max(masked_inputs, axis=-2)
        # return tf.clip_by_value(reduced_max, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)
        return reduced_max

    def get_config(self):
        return super(GlobalMaxPoolingRagged1D, self).get_config()


class Similarities(tkl.Layer):
    def __init__(self, has_weights=False, normalize=False, **kwargs):
        super(Similarities, self).__init__(**kwargs)
        self.supports_masking = True
        self.normalize = normalize
        self.has_weights = has_weights
        self._config_dict = {
            'normalize': normalize,
            'has_weights': has_weights
        }

    def compute_mask(self, inputs, mask=None):
        _, mmask = mask
        if len(mmask.shape) == 1:
            mmask = tf.expand_dims(mmask, axis=-2)
        assert len(mmask.shape) == 2
        return mmask

    def build(self, input_shape):
        oneshape, _ = input_shape
        param_shape = (oneshape[-1],)
        if self.has_weights:
            self.simdiag_weight = self.add_weight(name='sim_weight',
                                                  shape=param_shape,
                                                  initializer='glorot_uniform',  # 'zeros',  #
                                                  # regularizer=tk.regularizers.l2(1e-3),
                                                  trainable=True
                                                  )
        else:
            self.simdiag_weight = tf.ones(shape=param_shape, dtype=TF_DTYPE)
        super(Similarities, self).build(input_shape)

    def call(self, inputs, mask=None, training=None): # inputs = [sig, ancs]
        one, many = inputs
        if len(one.shape) == 2:
            one = tf.expand_dims(one, axis=-2)
        if len(many.shape) == 2:
            many = tf.expand_dims(many, axis=-2)
        assert len(one.shape) == len(many.shape) == 3

        if self.normalize:
            one = tf.math.l2_normalize(one, axis=-1)
            many = tf.math.l2_normalize(many, axis=-1)

        # tf.debugging.assert_all_finite(one, message='sim-pool: check query nan')
        # tf.debugging.assert_all_finite(many, message='sim-pool: check target nan')
        onew = tf.math.multiply(self.simdiag_weight, one)  # (..., 1, nChannels)
        sims = tf.matmul(many, onew, transpose_b=True)  # (..., seq_len_k, 1)
        # clipped_sims = tf.clip_by_value(sims, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)
        return sims

    def get_config(self):
        config = super(Similarities, self).get_config()
        config.update(self._config_dict)
        return config


class SimilaritiesDual(tkl.Layer):
    def __init__(self, has_weights=False, normalize=False, **kwargs):
        super(SimilaritiesDual, self).__init__(**kwargs)
        self.supports_masking = True
        self.normalize = normalize
        self.has_weights = has_weights
        self._config_dict = {
            'normalize': normalize,
            'has_weights': has_weights
        }

    def compute_mask(self, inputs, mask=None):
        _, mmask, dmask = mask
        return mmask, dmask

    def build(self, input_shape):
        oneshape, _, _ = input_shape
        param_shape = (oneshape[-1],)
        if self.has_weights:
            self.simdiag_weight = self.add_weight(name='sim_weight',
                                                  shape=param_shape,
                                                  initializer='glorot_uniform',  #'zeros',
                                                  trainable=True
                                                  )
        else:
            self.simdiag_weight = tf.ones(shape=param_shape, dtype=TF_DTYPE)
        super(SimilaritiesDual, self).build(input_shape)

    def call(self, inputs, mask=None, training=None): # inputs = [sig, attw-ancs, ancs]
        one, many, dual = inputs
        # print(one, many)
        assert len(one.shape) == len(many.shape)
        if self.normalize:
            one = tf.math.l2_normalize(one, axis=-1)
            many = tf.math.l2_normalize(many, axis=-1)
        onew = tf.math.multiply(self.simdiag_weight, one)  # (..., 1, nChannels)
        logits = tf.matmul(many, onew, transpose_b=True)  # (..., seq_len_k, 1)
        probs = tf.math.sigmoid(tf.matmul(dual, onew, transpose_b=True))
        return logits, probs

    def get_config(self):
        config = super(SimilaritiesDual, self).get_config()
        config.update(self._config_dict)
        return config


class SplitFirst(tkl.Layer):
    def __init__(self, squeeze_first=True, **kwargs):
        super(SplitFirst, self).__init__(**kwargs)
        self.supports_masking = True
        self.squeeze_first = squeeze_first
        self._config_dict = {
            'squeeze_first': squeeze_first
        }
        self.temp_masking = tkl.Masking(mask_value=0., name='split_first_tempmasking')

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = self._make_mask(inputs)
        mask_first = mask[:, :1]
        first_shape = (mask.shape[0],) + (1,) + mask.shape[2:]
        mask_first = tf.ensure_shape(mask_first, first_shape)
        if self.squeeze_first:
            ret_mask1 = tf.squeeze(mask_first, axis=1)
        else:
            ret_mask1 = mask_first
        return ret_mask1, mask[:, 1:]

    def call(self, inputs, mask=None, training=None):
        first = inputs[:, :1]
        first_shape = (first.shape[0],) + (1,) + first.shape[2:]
        first.set_shape(first_shape)
        rest = inputs[:, 1:]
        if self.squeeze_first:
            ret_first = tf.squeeze(first, axis=1)
        else:
            ret_first = first
        return ret_first, rest

    def get_config(self):
        config = super(SplitFirst, self).get_config()
        config.update(self._config_dict)
        return config

    def _make_mask(self, input):
        return self.temp_masking.compute_mask(input)


class CrossAttnPooling(tkl.Layer):
    def __init__(self, normalize=False, num_heads=None, kqweights_scale=False, kqweights_center=False,
                 squeeze_dims=None, **kwargs):
        super(CrossAttnPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.normalize = normalize
        self.num_heads = num_heads
        self.kqweights_scale = kqweights_scale
        self.kqweights_center = kqweights_center
        self.squeeze_dims = squeeze_dims
        self._config_dict = {
            'normalize': normalize,
            'num_heads': num_heads,
            'kqweights_scale': kqweights_scale,
            'kqweights_center': kqweights_center,
            'squeeze_dims': squeeze_dims
        }

    def compute_output_shape(self, input_shape):
        kshape, vshape, qshape = input_shape
        assert len(qshape) == len(kshape)
        # if len(qshape) < len(kshape):
        #     qshape = qshape[:-1] + (1, qshape[-1])
        if self.num_heads:
            out_fullshape = qshape[:2] + vshape[1:] # (batch_size, seq_len_q, seq_len_v, vdim)
            att_fullshape = qshape[:2] + (self.num_heads, vshape[1])  # (batch_size, sq(1), num_heads, seq_len_v,  sq(1))
            if self.squeeze_dims is None:
                outshape = out_fullshape
                attshape = att_fullshape
            else:
                osp = list(out_fullshape)
                outshape = tuple([s for j, s in enumerate(osp) if j not in self.squeeze_dims])
                asp = list(att_fullshape)
                attshape = tuple([a for i, a in enumerate(asp) if i not in self.squeeze_dims])
        else:
            attshape = kshape[:1] # (b,)
            outshape = vshape
        return attshape, outshape

    def compute_mask(self, inputs, mask=None):
        kmask, vmask, qmask = mask
        assert len(qmask.shape) == len(kmask.shape)
        # if len(qmask.shape) < len(kmask.shape):
        #     qmask = tf.expand_dims(qmask, axis=-1)
        if self.num_heads:
            qm = tf.expand_dims(qmask, axis=-1) # (batch_size, seq_len_q, 1)
            vm = tf.expand_dims(vmask, axis=-2) # (batch_size, 1, seq_len_v)
            out_fullmask = tf.math.logical_and(qm, vm) # (batch_size, seq_len_q, seq_len_v)
            attqm = tf.expand_dims(qm, axis=-1)
            attvm = tf.tile(tf.expand_dims(vm, axis=-2), multiples=(1, 1, self.num_heads, 1))
            att_fullmask = tf.math.logical_and(attqm, attvm)  # (batch_size, sq(1), num_heads, seq_len_v)
            if self.squeeze_dims is None:
                attmask = att_fullmask
                outmask = out_fullmask
            else:
                attmask = tf.squeeze(att_fullmask, axis=self.squeeze_dims)
                outmask = tf.squeeze(out_fullmask, axis=self.squeeze_dims)
        else:
            attmask = tf.math.reduce_any(kmask, axis=range(1, len(kmask.shape)))
            outmask = vmask
        return attmask, outmask

    def build(self, input_shape):
        kshape, _, _ = input_shape
        param_shape = (kshape[-1],)
        if self.kqweights_center:
            self.kqdiag_center = self.add_weight(name='kqdiag_center',
                                                 shape=param_shape,
                                                 initializer='zeros',
                                                 trainable=self.kqweights_center
                                                 )
        else:
            self.kqdiag_center = tf.zeros(shape=param_shape, dtype=TF_DTYPE) if self.kqweights_scale else tf.ones(shape=param_shape, dtype=TF_DTYPE)
            # self.kqdiag_center = self.add_weight(name='kqdiag_scala',
            #                                      initializer='glorot_uniform',
            #                                      trainable=True
            #                                      )
        if self.kqweights_scale:
            self.kqdiag_scale = self.add_weight(name='kqdiag_scale',
                                                shape=param_shape,
                                                initializer='ones',
                                                trainable=self.kqweights_scale
                                                )
        else:
            self.kqdiag_scale = tf.zeros(shape=param_shape, dtype=TF_DTYPE)
        super(CrossAttnPooling, self).build(input_shape)

    def call(self, inputs, mask=None, training=None): # inputs = [k,q]
        kmask, _, _ = mask
        if self.num_heads is None:
            _, value, _ = inputs # (b, t, v)
            num = tf.math.reduce_sum(tf.cast(kmask, dtype=TF_DTYPE), axis=-1, keepdims=True) # (b,1)
            weight = tf.math.reciprocal(tf.expand_dims(num, axis=-1)) # (b,1,1)
            out_tensor = tf.math.multiply(weight, value)
        else:
            key, value, query = inputs
            assert len(query.shape) == len(key.shape)
            _k = key
            _q = query
            _v = value
            weight, out_tensor = self._multiheads_attention_weighted(q=_q, k=_k, v=_v, mask=kmask) # (batch_size, seq_len_q, seq_len_v, d_model)

        if self.squeeze_dims is None or self.num_heads is None:
            return_tensor = out_tensor
            weight_sqdims = -1
        else:
            return_tensor = tf.squeeze(out_tensor, axis=self.squeeze_dims)  # (batch_size, seq_len_q, seq_len_v, d_model)
            weight_sqdims = self.squeeze_dims + [-1]
        return_weighting = tf.squeeze(weight, axis=weight_sqdims)
        if self.normalize:
            return return_weighting, tf.math.l2_normalize(return_tensor, axis=-1)
        return return_weighting, return_tensor

    def get_config(self):
        config = super(CrossAttnPooling, self).get_config()
        config.update(self._config_dict)
        return config

    def _multiheads_attention_weighted(self, q, k, v, mask=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.
        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.
        Returns:
        output, attention-weighted v
        """
        # pre-process
        d_model = q.shape[-1]
        vd = v.shape[-1]
        assert k.shape[-1] == d_model
        assert d_model % self.num_heads == 0 and vd % self.num_heads == 0
        depth = d_model // self.num_heads
        split = [self.num_heads, depth]
        vshape_new = tf.concat([tf.shape(q)[:2], tf.shape(v)[1:]], axis=-1)

        # multi-head splitting
        q = _split_heads(q, split_list=split)
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len_q, depth)
        k = _split_heads(k, split_list=split)
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len_k, depth)

        scale = tf.expand_dims(_split_heads(self.kqdiag_scale, split_list=split), axis=-2)
        offset = tf.expand_dims(_split_heads(self.kqdiag_center, split_list=split), axis=-2)

        v = tf.broadcast_to(tf.expand_dims(v, axis=1), shape=vshape_new)  # (batch_size, seq_len_q, seq_len_v, depth_v)
        vdep = vd // self.num_heads
        vsplit = [self.num_heads, vdep]
        v = _split_heads(v, split_list=vsplit)
        v = tf.transpose(v, perm=[0, 3, 1, 2, 4])  # (batch_size, num_heads, seq_len_q, seq_len_v, depth)

        if mask is not None:
            att_mask = mask[:, tf.newaxis, tf.newaxis, :]
            mask_id = tf.cast(att_mask, dtype=TF_DTYPE)

        kq_metric = tf.expand_dims(k, axis=-3)  # (..., 1, seq_len_k, depth)
        # kq_metric = tf.math.subtract(aligned_k, aligned_q) # (..., seq_len_q, seq_len_k, depth)
        # kq_metric = tf.math.multiply(aligned_k, aligned_q) # (..., seq_len_q, seq_len_k, depth)
        if mask is not None:
            mask_weight = tf.expand_dims(mask_id, axis=-1) + tf.zeros_like(kq_metric, dtype=TF_DTYPE)
        else:
            mask_weight = tf.ones_like(kq_metric, dtype=TF_DTYPE)
        mask_weight.set_shape(kq_metric.shape)

        mean_metric, var_metric = _masked_moments(kq_metric, mask_identity=mask_weight, axis=-2, keep_dims=False)
        std = tf.math.sqrt(var_metric)  # (..., seq_len_q, depth)
        sigma_ratio = tf.math.multiply(std, tf.math.reciprocal_no_nan(tf.math.reduce_max(std, axis=-1, keepdims=True)))
        _diag = tf.math.add(offset, tf.math.multiply(scale, sigma_ratio))
        diag_q = tf.math.multiply(q, _diag)  # (..., seq_len_q, depth)

        # scaled dot product by head
        matmul_qk = tf.matmul(diag_q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        scaling_factor = tf.cast(depth, TF_DTYPE)
        scaled_attention_logits = tf.multiply(matmul_qk, tf.math.reciprocal(tf.math.sqrt(scaling_factor)))

        # add the mask to the scaled attention
        if mask is not None:
            adder = (1. - mask_id) * _LARGE_NEG(TF_DTYPE)  # (..., 1, seq_len_k)
            # broadcast to (..., seq_len_q, seq_len_k)
            attn_logits = tf.math.add(tf.math.multiply(scaled_attention_logits, mask_id), adder)
        else:
            attn_logits = scaled_attention_logits
        # softmax normalization and matching v.shape
        attention_weights = tf.expand_dims(tf.math.softmax(attn_logits, axis=-1), axis=-1)  # (..., seq_len_q, seq_len_k, 1)
        assert len(attention_weights.shape) == len(v.shape)  # same rank

        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_v, 1)
        # implicit broadcast
        weighted_v = tf.math.multiply(attention_weights, v)  # (..., seq_len_q, seq_len_v, depth_v)
        # if mask is not None:
        #     vmask = tf.expand_dims(mask_id, axis=-1) + tf.zeros_like(v, dtype=TF_DTYPE)
        # else:
        #     vmask = tf.ones_like(v, dtype=TF_DTYPE)
        # vmask.set_shape(v.shape)
        # _, vvar = _masked_moments(v, mask_identity=vmask, axis=-2, keep_dims=True)
        # vstd = tf.math.sqrt(vvar)  # (..., seq_len_q, 1, depth)
        # v_ratio = tf.math.multiply(vstd, tf.math.reciprocal_no_nan(tf.math.reduce_max(vstd, axis=-1, keepdims=True)))
        # print('V:', vmask, v_ratio)
        # weighted_v = tf.math.multiply(tf.math.multiply(attention_weights, v_ratio), v)

        backscaled_v = tf.transpose(weighted_v, perm=[0, 2, 3, 1, 4])  # (batch_size, seq_len_q, seq_len_v, num_heads, vdep)
        heads_concat = tf.reshape(backscaled_v, vshape_new)  # (batch_size, seq_len_q, seq_len_v, vd)
        attention_scores = tf.transpose(attention_weights, perm=[0, 2, 1, 3, 4])  # (batch_size, seq_len_q, num_heads, seq_len_k,  1)

        return attention_scores, heads_concat


class CrossDistAttnPooling(tkl.Layer):
    def __init__(self, l1_distance=True, kqweights=False, use_bias=True,
                 num_heads=None, squeeze_dims=None, **kwargs):
        super(CrossDistAttnPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_heads = num_heads
        self.squeeze_dims = squeeze_dims
        self.kqweights = kqweights
        self.use_bias = use_bias
        self.L1_distance = l1_distance
        self._config_dict = {
            'num_heads': num_heads,
            'squeeze_dims': squeeze_dims,
            'kqweights': kqweights,
            'use_bias': use_bias,
            'l1_distance': l1_distance
        }

    def compute_output_shape(self, input_shape):
        kshape, vshape, qshape = input_shape
        assert len(qshape) == len(kshape)
        # if len(qshape) < len(kshape):
        #     qshape = qshape[:-1] + (1, qshape[-1])
        if self.num_heads:
            out_fullshape = qshape[:2] + vshape[1:]  # (batch_size, seq_len_q, seq_len_v, vdim)
            att_fullshape = qshape[:2] + (
            self.num_heads, vshape[1])  # (batch_size, sq(1), num_heads, seq_len_v,  sq(1))
            if self.squeeze_dims is None:
                outshape = out_fullshape
                attshape = att_fullshape
            else:
                osp = list(out_fullshape)
                outshape = tuple([s for j, s in enumerate(osp) if j not in self.squeeze_dims])
                asp = list(att_fullshape)
                attshape = tuple([a for i, a in enumerate(asp) if i not in self.squeeze_dims])
        else:
            attshape = kshape[:1]  # (b,)
            outshape = vshape
        return attshape, outshape

    def compute_mask(self, inputs, mask=None):
        kmask, vmask, qmask = mask
        assert len(qmask.shape) == len(kmask.shape)
        # if len(qmask.shape) < len(kmask.shape):
        #     qmask = tf.expand_dims(qmask, axis=-1)
        if self.num_heads:
            qm = tf.expand_dims(qmask, axis=-1)  # (batch_size, seq_len_q, 1)
            vm = tf.expand_dims(vmask, axis=-2)  # (batch_size, 1, seq_len_v)
            out_fullmask = tf.math.logical_and(qm, vm)  # (batch_size, seq_len_q, seq_len_v)
            attqm = tf.expand_dims(qm, axis=-1)
            attvm = tf.tile(tf.expand_dims(vm, axis=-2), multiples=(1, 1, self.num_heads, 1))
            att_fullmask = tf.math.logical_and(attqm, attvm)  # (batch_size, sq(1), num_heads, seq_len_v)
            if self.squeeze_dims is None:
                attmask = att_fullmask
                outmask = out_fullmask
            else:
                attmask = tf.squeeze(att_fullmask, axis=self.squeeze_dims)
                outmask = tf.squeeze(out_fullmask, axis=self.squeeze_dims)
            # print('!!!', att_fullmask, attmask, outmask)
        else:
            attmask = tf.math.reduce_any(kmask, axis=range(1, len(kmask.shape)))
            outmask = vmask
        return attmask, outmask

    def build(self, input_shape):
        kshape, _, _ = input_shape
        vec_size = kshape[-1]
        gate_depth = vec_size // self.num_heads
        gate_shape = (self.num_heads, gate_depth)
        # init_eye = np.tile(np.expand_dims(np.eye(gate_depth, dtype=DTYPE), axis=-1), (self.num_heads, 1, 1))
        if self.kqweights:
            self.kernel = self.add_weight(name='linear_weights',
                                          shape=gate_shape + (gate_depth,),
                                          initializer='glorot_uniform'  # tk.initializers.Constant(init_eye)  # 'zeros'  #
                                          )
            if self.use_bias:
                self.bias = self.add_weight(name='linear_bias',
                                            shape=gate_shape,
                                            initializer='zeros'
                                            )
            else:
                self.bias = tf.zeros(shape=gate_shape, dtype=TF_DTYPE)
        else:
            self.kernel = None
            if self.use_bias:
                self.bias = self.add_weight(name='linear_bias',
                                            shape=gate_shape,
                                            initializer='ones'
                                            )
            else:
                self.bias = tf.ones(shape=gate_shape, dtype=TF_DTYPE)
        super(CrossDistAttnPooling, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):  # inputs = [k,v,q]
        kmask, _, _ = mask
        if self.num_heads is None:
            _, value, _ = inputs  # (b, t, v)
            num = tf.math.reduce_sum(tf.cast(kmask, dtype=TF_DTYPE), axis=-1, keepdims=True)  # (b,1)
            weight = tf.math.reciprocal(tf.expand_dims(num, axis=-1))  # (b,1,1)
            out_tensor = tf.math.multiply(weight, value)
        else:
            key, value, query = inputs
            assert len(query.shape) == len(key.shape)
            # if len(query.shape) < len(key.shape):
            #     qry = tf.expand_dims(query, axis=-2)
            # else:
            #     qry = query
            _k = key
            _q = query
            _v = value

            # weight, out_tensor = _multiheads_distattn_weighted(q=_q, k=_k, v=_v,
            #                                                    params_weights=self.kqdiag_weight,
            #                                                    params_bias=self.kqdiag_bias,
            #                                                    num_heads=self.num_heads,
            #                                                    mask=kmask
            #                                                    ) # (batch_size, seq_len_q, seq_len_v, d_model)
            weight, out_tensor = self._multiheads_distattnse_weighted(q=_q, k=_k, v=_v, mask=kmask)

        if self.squeeze_dims is None:
            return_tensor = out_tensor
            weight_sqdims = -1
        else:
            return_tensor = tf.squeeze(out_tensor, axis=self.squeeze_dims)  # sq (batch_size, seq_len_q, seq_len_v, d_model)
            weight_sqdims = self.squeeze_dims + [-1, ]
        return_weighting = tf.squeeze(weight, axis=weight_sqdims)
        return return_weighting, return_tensor

    def get_config(self):
        config = super(CrossDistAttnPooling, self).get_config()
        config.update(self._config_dict)
        return config

    def _multiheads_distattnse_weighted(self, q, k, v, mask=None, eps=1e-9):
        # pre-process
        d_model = q.shape[-1]
        vd = v.shape[-1]
        assert k.shape[-1] == d_model
        assert d_model % self.num_heads == 0 and vd % self.num_heads == 0
        depth = d_model // self.num_heads
        split = [self.num_heads, depth]
        vshape_new = tf.concat([tf.shape(q)[:2], tf.shape(v)[1:]], axis=-1)

        # dist-based by head
        aligned_k = tf.expand_dims(k, axis=-3)  # (..., 1, seq_len_k, ch)
        aligned_q = tf.expand_dims(q, axis=-2)  # (..., seq_len_q, 1, ch)
        diff = tf.math.subtract(aligned_k, aligned_q)  # (..., seq_len_q, seq_len_k, ch)
        mh_diff = _split_heads(diff, split_list=split)  # (b, q, k, h, dep)
        _nCh = tf.cast(depth, TF_DTYPE)
        # MH Squeeze, broadcast to (..., seq_len_q, seq_len_k, ch)
        if self.L1_distance:
            # metric = tf.math.abs(diff)
            metric = aligned_k
            _distance = tf.math.abs(mh_diff)
            scaling_factor = tf.math.sqrt(_nCh * 0.72676)
            center_factor = _nCh * 1.12838
        else:
            metric = tf.math.square(diff)
            _distance = tf.math.square(mh_diff)
            scaling_factor = tf.math.sqrt(_nCh * 8.)
            center_factor = _nCh * 2.

        if mask is not None:
            att_mask = mask[:, tf.newaxis, :]  # (b, 1, seq_len_k)
            masked1s = tf.cast(att_mask, dtype=TF_DTYPE)
            mask_weights = tf.expand_dims(masked1s, axis=-1) + tf.zeros_like(metric, dtype=TF_DTYPE)  # (..., q, k, ch)
        else:
            mask_weights = tf.ones_like(metric, dtype=TF_DTYPE)
        mask_weights.set_shape(metric.shape)  # (..., seq_len_q, seq_len_k, ch)
        mean_metric, var_metric = _masked_moments(metric, mask_identity=mask_weights, axis=-2,
                                                  keep_dims=True)  # (..., seq_len_q, 1, ch)
        std_metric = tf.math.sqrt(var_metric)

        # std_metric = _split_heads(std_metric, split_list=split)
        # max_std = tf.math.reduce_max(std_metric, axis=-1, keepdims=True)
        # sqz = tf.math.subtract(tf.math.log(std_metric), tf.math.log(max_std + eps - std_metric))

        sqz_metric = tf.math.log(std_metric + eps)
        sqz = _split_heads(sqz_metric, split_list=split)  # (..., seq_len_q, 1(k), h, dep)
        # print(sqz)

        # Excite
        if self.kqweights:
            _xw = tf.einsum('...hd,hdc->...hc', sqz, self.kernel)
            _se = tf.math.sigmoid(tf.math.add(_xw, self.bias))
            _dist = tf.math.multiply(_se, _distance)
        else:
            _dist = tf.math.multiply(self.bias, _distance)

        dist = tf.transpose(_dist, perm=[0, 3, 1, 2, 4])  # (batch_size, num_heads, seq_len_q, k, depth)
        distance = tf.math.reduce_sum(dist, axis=-1, keepdims=False)  # (..., seq_len_q, seq_len_k)
        scaled_attention = tf.math.multiply(tf.math.subtract(center_factor, distance),
                                            tf.math.reciprocal(scaling_factor))
        # scaled_attention = tf.math.add(safe_divide(numerator=scaling_factor, denominator=distance, epsilon=eps), eps)
        if mask is not None:
            # add the mask to the scaled attention
            attn_mask = tf.expand_dims(masked1s, axis=-3)  # mask's head-dim after transpose: -3
            # masked_attn = attn_mask * scaled_attention  # broadcast to (..., seq_len_q, seq_len_k)
            _neg_inf = _LARGE_NEG(TF_DTYPE)
            masked_attn = tf.math.add(tf.math.multiply(attn_mask, scaled_attention), (1. - attn_mask) * _neg_inf)
            # print(attn_mask, scaled_attention)
        else:
            masked_attn = scaled_attention
        # normalization and matching v.shape
        # revised_softmax, _ = tf.linalg.normalize(masked_attn, ord=1, axis=-1)
        # attention_weights = tf.expand_dims(revised_softmax, axis=-1)  # (..., seq_len_q, seq_len_k, 1)
        attention_weights = tf.expand_dims(tf.math.softmax(masked_attn, axis=-1), axis=-1)

        v = tf.broadcast_to(tf.expand_dims(v, axis=1), shape=vshape_new)  # (batch_size, seq_len_q, seq_len_v, depth_v)
        vdep = vd // self.num_heads
        vsplit = [self.num_heads, vdep]
        v = _split_heads(v, split_list=vsplit)
        v = tf.transpose(v, perm=[0, 3, 1, 2, 4])  # (batch_size, num_heads, seq_len_q, seq_len_v, depth)

        assert len(attention_weights.shape) == len(v.shape)  # same rank
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_v, 1), implicit broadcast
        weighted_v = tf.math.multiply(attention_weights, v)  # (..., seq_len_q, seq_len_v, depth_v)
        backscaled_v = tf.transpose(weighted_v,
                                    perm=[0, 2, 3, 1, 4])  # (batch_size, seq_len_q, seq_len_v, num_heads, vdep)
        heads_concat = tf.reshape(backscaled_v, vshape_new)  # (batch_size, seq_len_q, seq_len_v, vd)
        attention_scores = tf.transpose(attention_weights,
                                        perm=[0, 2, 1, 3, 4])  # (batch_size, seq_len_q, num_heads, seq_len_k,  1)

        return attention_scores, heads_concat


class MSALayer(tkl.Layer):
    def __init__(self,
                 transformer_config,
                 vec_size,
                 num_layers=12,
                 return_final_encoder=False,
                 return_all_encoder_outputs=False,
                 trf_size=None,
                 trf_actv=None,
                 pooler_size=None,
                 pooler_actv=None,
                 pre_layernorm=False,
                 pooler_normalize=False, # True,
                 position_embeddings=True,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 activation=activations.gelu,
                 **kwargs):
        super(MSALayer, self).__init__(**kwargs)
        self.supports_masking = True
        assert not (return_final_encoder and return_all_encoder_outputs)
        actv = tk.activations.get(activation)
        init = tk.initializers.get(initializer)
        self._config_dict = {
            'transformer_config': transformer_config, # json.dumps(transformer_config, indent=4) # import json
            'vec_size': vec_size,
            'num_layers': num_layers,
            'return_final_encoder': return_final_encoder,
            'return_all_encoder_outputs': return_all_encoder_outputs,
            'trf_size': trf_size,
            'trf_actv': trf_actv,
            'pooler_size': pooler_size,
            'pooler_actv': pooler_actv,
            'pre_layernorm': pre_layernorm,
            'pooler_normalize': pooler_normalize,
            'position_embeddings': position_embeddings,
            'initializer': tf.keras.initializers.serialize(init),
            'activation': tf.keras.activations.serialize(actv) # 'gelu'
        }
        self.tconfig = transformer_config
        self.vec_size = vec_size
        self.trf_size = trf_size
        self.pooler_size = pooler_size
        if self.trf_size is not None:
            if trf_actv is None:
                tactv = 'relu' if self.trf_size > self.vec_size else 'tanh'
            else:
                tactv = trf_actv
            self.input_transform = tk.layers.Dense(
                units=self.trf_size,
                activation=tactv,
                kernel_initializer=init,
                # kernel_regularizer='l2',
                name='input_transform')
        self.max_seqlen = transformer_config['max_sequence_length']
        self.pooler_normalize = pooler_normalize
        self.return_final_encoder = return_final_encoder
        self.return_all_encoder_outputs = return_all_encoder_outputs
        self.position_embedding_layer = None
        if position_embeddings:
            self.position_embedding_layer = PositionEmbedding(
                # initializer=init,
                use_dynamic_slicing=True,
                max_length=self.max_seqlen,
                seq_axis=-2,
                name='position_embedding')
        self.add_layer = tkl.Add()
        if pre_layernorm:
            self.normalize_layer = tkl.LayerNormalization(
                name='embeddings/layer_norm',
                axis=-1,
                epsilon=1e-6,
                dtype=TF_DTYPE
            )
        else:
            self.normalize_layer = None
        self.dropout_layer = tkl.Dropout(rate=self.tconfig['dropout_rate'])
        self.sa_mask = SelfAttentionMask()
        self.tlayers = []
        for i in range(num_layers):
            layer = Transformer(
                num_attention_heads=transformer_config['num_attention_heads'],
                intermediate_size=transformer_config['intermediate_size'],
                intermediate_activation=actv,
                dropout_rate=transformer_config['dropout_rate'],
                attention_dropout_rate=transformer_config['attention_dropout_rate'],
                kernel_initializer=init,
                name='transformer/layer_%d' % i)
            self.tlayers.append(layer)
        self.pooler_layer = tkl.Lambda(lambda _x: tf.squeeze(_x[:, 0:1, :], axis=1))
        if self.pooler_normalize:
            self.pooler_normlayer = tkl.Lambda(lambda _x: tf.math.l2_normalize(_x, axis=-1))
        else:
            self.pooler_normlayer = None
        if self.pooler_size is not None:
            if pooler_actv is None:
                input_size = self.trf_size if self.trf_size is not None else self.vec_size
                pactv = 'relu' if self.pooler_size > input_size else 'tanh'
            else:
                pactv = pooler_actv
            self.pooler_transform = tk.layers.Dense(
                units=self.pooler_size,
                activation=pactv,
                kernel_initializer=init,
                # kernel_regularizer='l2',
                name='pooler_transform')

    # Must have for use in e.g. H1AttnLayer
    def compute_output_shape(self, input_shape):
        if self.return_final_encoder or self.return_all_encoder_outputs:
            return input_shape
        last_dim = self.pooler_size if self.pooler_size is not None else self.vec_size
        return input_shape[:-2]+(last_dim,)

    def compute_mask(self, inputs, mask=None):
        if mask is None or self.return_final_encoder or self.return_all_encoder_outputs:
            return mask
        return tf.math.reduce_any(mask, axis=-1)

    def call(self, inputs, mask=None, training=None):
        combined_mask = None
        inputs_vec = inputs
        if mask is not None:
            combined_mask = mask
        if self.trf_size is not None:
            inputs_vec = self.input_transform(inputs_vec)
        # print('AttnLayer input shape =', inputs_vec.shape) # shape = (batch, seq_length, vec_size)
        if self.position_embedding_layer: # is not None:
            pe = self.position_embedding_layer(inputs_vec)
            embeddings = self.add_layer([inputs_vec, pe])
        else:
            embeddings = inputs_vec
        if self.normalize_layer is not None:
            embeddings = self.normalize_layer(embeddings)
        if training:
            embeddings = self.dropout_layer(embeddings)
        x = embeddings
        if combined_mask is not None:
            attention_mask = self.sa_mask([x, combined_mask])
        else:
            print('No attention_mask')
            attention_mask = None
        encoder_outputs = []
        for tlayer in self.tlayers:
            if attention_mask is not None:
                x = tlayer([x, attention_mask])
            else:
                x = tlayer(x)
            encoder_outputs.append(x)
        if self.return_all_encoder_outputs:
            return encoder_outputs
        elif self.return_final_encoder:
            return encoder_outputs[-1] # [-1] is the final layer of all positions
        else:
            y = self.pooler_layer(encoder_outputs[-1])
            if self.pooler_size is not None:
                y = self.pooler_transform(y)
            if self.pooler_normalize:
                y = self.pooler_normlayer(y)
            return y

    def get_config(self):
        config = super(MSALayer, self).get_config()
        config.update(self._config_dict)
        return config


class MSALayerAS(tkl.Layer):
    def __init__(self,
                 transformer_config,
                 vec_size,
                 num_layers=12,
                 return_final_encoder=False,
                 return_all_encoder_outputs=False,
                 trf_size=None,
                 trf_actv=None,
                 pooler_size=None,
                 pooler_actv=None,
                 pre_layernorm=False,
                 pooler_normalize=False, # True,
                 position_embeddings=True,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 activation=activations.gelu,
                 **kwargs):
        super(MSALayerAS, self).__init__(**kwargs)
        self.supports_masking = True
        assert not (return_final_encoder and return_all_encoder_outputs)
        actv = tk.activations.get(activation)
        init = tk.initializers.get(initializer)
        self._config_dict = {
            'transformer_config': transformer_config, # json.dumps(transformer_config, indent=4) # import json
            'vec_size': vec_size,
            'num_layers': num_layers,
            'return_final_encoder': return_final_encoder,
            'return_all_encoder_outputs': return_all_encoder_outputs,
            'trf_size': trf_size,
            'trf_actv': trf_actv,
            'pooler_size': pooler_size,
            'pooler_actv': pooler_actv,
            'pre_layernorm': pre_layernorm,
            'pooler_normalize': pooler_normalize,
            'position_embeddings': position_embeddings,
            'initializer': tf.keras.initializers.serialize(init),
            'activation': tf.keras.activations.serialize(actv) # 'gelu'
        }
        self.tconfig = transformer_config
        self.vec_size = vec_size
        self.trf_size = trf_size
        self.pooler_size = pooler_size
        if self.trf_size is not None:
            if trf_actv is None:
                tactv = 'relu' if self.trf_size > self.vec_size else 'tanh'
            else:
                tactv = trf_actv
            self.input_transform = tk.layers.Dense(
                units=self.trf_size,
                activation=tactv,
                kernel_initializer=init,
                # kernel_regularizer='l2',
                name='input_transform')
        self.max_seqlen = transformer_config['max_sequence_length']
        self.pooler_normalize = pooler_normalize
        self.return_final_encoder = return_final_encoder
        self.return_all_encoder_outputs = return_all_encoder_outputs
        self.position_embedding_layer = None
        if position_embeddings:
            self.position_embedding_layer = PositionEmbedding(
                # initializer=init,
                use_dynamic_slicing=True,
                max_length=self.max_seqlen,
                seq_axis=-2,
                name='position_embedding')
        self.add_layer = tkl.Add()
        if pre_layernorm:
            self.normalize_layer = tkl.LayerNormalization(
                name='embeddings/layer_norm',
                axis=-1,
                epsilon=1e-6,
                dtype=TF_DTYPE
            )
        else:
            self.normalize_layer = None
        self.dropout_layer = tkl.Dropout(rate=self.tconfig['dropout_rate'])
        self.sa_mask = SelfAttentionMask()
        self.tlayers = []
        for i in range(num_layers):
            layer = TransformerEncoderBlockAS(
                num_attention_heads=transformer_config['num_attention_heads'],
                inner_dim=transformer_config['intermediate_size'],
                inner_activation=actv,
                output_dropout=transformer_config['dropout_rate'],
                attention_dropout=transformer_config['attention_dropout_rate'],
                kernel_initializer=init,
                return_attention_scores=True,
                name='transformer/layer_%d' % i)
            self.tlayers.append(layer)
        self.pooler_layer = tkl.Lambda(lambda _x: tf.squeeze(_x[:, 0:1, :], axis=1))
        if self.pooler_normalize:
            self.pooler_normlayer = tkl.Lambda(lambda _x: tf.math.l2_normalize(_x, axis=-1))
        else:
            self.pooler_normlayer = None
        if self.pooler_size is not None:
            if pooler_actv is None:
                input_size = self.trf_size if self.trf_size is not None else self.vec_size
                pactv = 'relu' if self.pooler_size > input_size else 'tanh'
            else:
                pactv = pooler_actv
            self.pooler_transform = tk.layers.Dense(
                units=self.pooler_size,
                activation=pactv,
                kernel_initializer=init,
                # kernel_regularizer='l2',
                name='pooler_transform')

    # Must have for use in e.g. H1AttnLayer
    def compute_output_shape(self, input_shape):
        if self.return_final_encoder or self.return_all_encoder_outputs:
            return input_shape
        last_dim = self.pooler_size if self.pooler_size is not None else self.vec_size
        return input_shape[:-2]+(last_dim,)

    def compute_mask(self, inputs, mask=None):
        if mask is None or self.return_final_encoder or self.return_all_encoder_outputs:
            return mask
        return tf.math.reduce_any(mask, axis=-1)

    def call(self, inputs, mask=None, training=None):
        combined_mask = None
        inputs_vec = inputs
        if mask is not None:
            combined_mask = mask
        if self.trf_size is not None:
            inputs_vec = self.input_transform(inputs_vec)
        # print('AttnLayer input shape =', inputs_vec.shape) # shape = (batch, seq_length, vec_size)
        if self.position_embedding_layer: # is not None:
            pe = self.position_embedding_layer(inputs_vec)
            embeddings = self.add_layer([inputs_vec, pe])
        else:
            embeddings = inputs_vec
        if self.normalize_layer is not None:
            embeddings = self.normalize_layer(embeddings)
        if training:
            embeddings = self.dropout_layer(embeddings)
        x = embeddings
        if combined_mask is not None:
            attention_mask = self.sa_mask([x, combined_mask])
        else:
            print('No attention_mask')
            attention_mask = None
        encoder_outputs = []
        encoder_as = []
        for tlayer in self.tlayers:
            if attention_mask is not None:
                x, atts = tlayer([x, attention_mask])
            else:
                x, atts = tlayer(x)
            encoder_outputs.append(x)
            encoder_as.append(atts)
        if self.return_all_encoder_outputs:
            return encoder_outputs, encoder_as
        elif self.return_final_encoder:
            return encoder_outputs[-1], encoder_as[-1]  # [-1] is the final layer of all positions
        else:
            y = self.pooler_layer(encoder_outputs[-1])
            if self.pooler_size is not None:
                y = self.pooler_transform(y)
            if self.pooler_normalize:
                y = self.pooler_normlayer(y)
            return y, encoder_as[-1]

    def get_config(self):
        config = super(MSALayerAS, self).get_config()
        config.update(self._config_dict)
        return config


class TransformerEncoderBlockAS(tf.keras.layers.Layer):
    def __init__(self,
                 num_attention_heads,
                 inner_dim,
                 inner_activation,
                 output_range=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bias=True,
                 norm_first=False,
                 norm_epsilon=1e-12,
                 output_dropout=0.0,
                 attention_dropout=0.0,
                 inner_dropout=0.0,
                 attention_initializer=None,
                 attention_axes=None,
                 use_query_residual=True,
                 key_dim=None,
                 value_dim=None,
                 output_last_dim=None,
                 diff_q_kv_att_layer_norm=False,
                 return_attention_scores=False,
                 **kwargs):
        util.filter_kwargs(kwargs)
        super().__init__(**kwargs)

        # Deprecation warning.
        if output_range is not None:
            logging.warning("`output_range` is available as an argument for `call()`."
                            "The `output_range` as __init__ argument is deprecated.")

        self._num_heads = num_attention_heads
        self._inner_dim = inner_dim
        self._inner_activation = inner_activation
        self._attention_dropout_rate = attention_dropout
        self._output_dropout_rate = output_dropout
        self._output_range = output_range
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._inner_dropout = inner_dropout
        self._use_query_residual = use_query_residual
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._output_last_dim = output_last_dim
        self._diff_q_kv_att_layer_norm = diff_q_kv_att_layer_norm
        self._return_attention_scores = return_attention_scores
        if attention_initializer:
            self._attention_initializer = tf.keras.initializers.get(
                attention_initializer)
        else:
            self._attention_initializer = tf_utils.clone_initializer(
                self._kernel_initializer)
        self._attention_axes = attention_axes

        if self._diff_q_kv_att_layer_norm and not self._norm_first:
            raise ValueError("Setting `diff_q_and_kv_attention_layer_norm` to True"
                             "when `norm_first` is False is invalid.")

    def build(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_tensor_shape = input_shape
        elif isinstance(input_shape, (list, tuple)):
            input_tensor_shape = tf.TensorShape(input_shape[0])
        else:
            raise ValueError(
                "The type of input shape argument is not supported, got: %s" %
                type(input_shape))
        einsum_equation = "abc,cd->abd"
        if len(input_tensor_shape.as_list()) > 3:
            einsum_equation = "...bc,cd->...bd"
        hidden_size = input_tensor_shape[-1]
        if hidden_size % self._num_heads != 0:
            logging.warning(
                "The input size (%d) is not a multiple of the number of attention "
                "heads (%d)", hidden_size, self._num_heads)
        if self._key_dim is None:
            self._key_dim = int(hidden_size // self._num_heads)
        if self._output_last_dim is None:
            last_output_shape = hidden_size
        else:
            last_output_shape = self._output_last_dim

        common_kwargs = dict(
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint)
        self._attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._key_dim,
            value_dim=self._value_dim,
            dropout=self._attention_dropout_rate,
            use_bias=self._use_bias,
            kernel_initializer=self._attention_initializer,
            bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
            attention_axes=self._attention_axes,
            output_shape=self._output_last_dim,
            name="self_attention",
            **common_kwargs)
        self._attention_dropout = tf.keras.layers.Dropout(
            rate=self._attention_dropout_rate)
        # Use float32 in layernorm for numeric stability.
        # It is probably safe in mixed_float16, but we haven't validated this yet.
        self._attention_layer_norm = (
            tf.keras.layers.LayerNormalization(
                name="self_attention_layer_norm",
                axis=-1,
                epsilon=self._norm_epsilon,
                dtype=TF_DTYPE))
        self._attention_layer_norm_kv = self._attention_layer_norm
        if self._diff_q_kv_att_layer_norm:
            self._attention_layer_norm_kv = (
                tf.keras.layers.LayerNormalization(
                    name="self_attention_layer_norm_kv",
                    axis=-1,
                    epsilon=self._norm_epsilon,
                    dtype=TF_DTYPE))

        self._intermediate_dense = tf.keras.layers.EinsumDense(
            einsum_equation,
            output_shape=(None, self._inner_dim),
            bias_axes="d",
            kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
            bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
            name="intermediate",
            **common_kwargs)
        policy = tf.keras.mixed_precision.global_policy()
        if policy.name == "mixed_bfloat16":
            policy = tf.float32
        self._intermediate_activation_layer = tf.keras.layers.Activation(
            self._inner_activation, dtype=policy)
        self._inner_dropout_layer = tf.keras.layers.Dropout(
            rate=self._inner_dropout)
        self._output_dense = tf.keras.layers.EinsumDense(
            einsum_equation,
            output_shape=(None, last_output_shape),
            bias_axes="d",
            name="output",
            kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
            bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
            **common_kwargs)
        self._output_dropout = tf.keras.layers.Dropout(
            rate=self._output_dropout_rate)
        # Use float32 in layernorm for numeric stability.
        self._output_layer_norm = tf.keras.layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=TF_DTYPE)

        super().build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads":
                self._num_heads,
            "inner_dim":
                self._inner_dim,
            "inner_activation":
                self._inner_activation,
            "output_dropout":
                self._output_dropout_rate,
            "attention_dropout":
                self._attention_dropout_rate,
            "output_range":
                self._output_range,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint":
                tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint":
                tf.keras.constraints.serialize(self._bias_constraint),
            "use_bias":
                self._use_bias,
            "norm_first":
                self._norm_first,
            "norm_epsilon":
                self._norm_epsilon,
            "inner_dropout":
                self._inner_dropout,
            "attention_initializer":
                tf.keras.initializers.serialize(self._attention_initializer),
            "attention_axes":
                self._attention_axes,
            "use_query_residual":
                self._use_query_residual,
            "key_dim":
                self._key_dim,
            "value_dim":
                self._value_dim,
            "output_last_dim":
                self._output_last_dim,
            "diff_q_kv_att_layer_norm":
                self._diff_q_kv_att_layer_norm,
            'return_attention_scores':
                self._return_attention_scores
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: Any, output_range: Optional[tf.Tensor] = None) -> Any:
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                input_tensor, attention_mask = inputs
                key_value = None
            elif len(inputs) == 3:
                input_tensor, key_value, attention_mask = inputs
            else:
                raise ValueError("Unexpected inputs to %s with length at %d" %
                                 (self.__class__, len(inputs)))
        else:
            input_tensor, key_value, attention_mask = (inputs, None, None)

        if output_range is None:
            output_range = self._output_range
        if output_range:
            if self._norm_first:
                source_tensor = input_tensor[:, 0:output_range, :]
                input_tensor = self._attention_layer_norm(input_tensor)
                if key_value is not None:
                    key_value = self._attention_layer_norm_kv(key_value)
            target_tensor = input_tensor[:, 0:output_range, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 0:output_range, :]
        else:
            if self._norm_first:
                source_tensor = input_tensor
                input_tensor = self._attention_layer_norm(input_tensor)
                if key_value is not None:
                    key_value = self._attention_layer_norm_kv(key_value)
            target_tensor = input_tensor

        if key_value is None:
            key_value = input_tensor

        if self._return_attention_scores:
            attention_output, attention_scores = self._attention_layer(
                query=target_tensor,
                value=key_value,
                attention_mask=attention_mask,
                return_attention_scores=True)
        else:
            attention_output = self._attention_layer(
                query=target_tensor, value=key_value, attention_mask=attention_mask)
        attention_output = self._attention_dropout(attention_output)

        if self._norm_first:
            # Important to not combine `self._norm_first` and
            # `self._use_query_residual` into one if clause because else is only for
            # `_norm_first == False`.
            if self._use_query_residual:
                attention_output = source_tensor + attention_output
        else:
            if self._use_query_residual:
                attention_output = target_tensor + attention_output
            attention_output = self._attention_layer_norm(attention_output)

        if self._norm_first:
            source_attention_output = attention_output
            attention_output = self._output_layer_norm(attention_output)
        inner_output = self._intermediate_dense(attention_output)
        inner_output = self._intermediate_activation_layer(inner_output)
        inner_output = self._inner_dropout_layer(inner_output)
        layer_output = self._output_dense(inner_output)
        layer_output = self._output_dropout(layer_output)

        if self._norm_first:
            layer_output = source_attention_output + layer_output
        else:
            # During mixed precision training, layer norm output is always fp32 for
            # now. Casts fp32 for the subsequent add.
            layer_output = tf.cast(layer_output, tf.float32)
            layer_output = self._output_layer_norm(layer_output + attention_output)

        if self._return_attention_scores:
            return layer_output, attention_scores
        else:
            return layer_output


class MilAttn(tkl.Layer):
    def __init__(self, hidden_dim=128, gated_attn=True, extention=False, **kwargs):
        super(MilAttn, self).__init__(**kwargs)
        self.supports_masking = True
        self.hidden_dim = hidden_dim
        self.gated_attn = gated_attn
        self.extention = extention
        self._config_dict = {
            'hidden_dim': hidden_dim,
            'gated_attn': gated_attn,
            'extention': extention
        }

    def build(self, input_shape):
        assert len(input_shape) == 3 # batch, time_seq, vec_size
        if self.extention:
            vec_size = input_shape[-1] * 4
        else:
            vec_size = input_shape[-1]
        self.V = self.add_weight(name='V',
                                 shape=(vec_size, self.hidden_dim),
                                 initializer='glorot_uniform',
                                 trainable=True
                                 )
        self.w = self.add_weight(name='w',
                                 shape=(self.hidden_dim, 1),
                                 initializer='glorot_uniform',
                                 trainable=True
                                 )
        self.U = self.add_weight(name='U',
                                 shape=(vec_size, self.hidden_dim),
                                 initializer='glorot_uniform',
                                 trainable=True
                                 )
        super(MilAttn, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.extention:
            fst_dim = input_shape[0]
            snd_dim = input_shape[1]-1
            inshape = (fst_dim, snd_dim) + input_shape[2:]
        else:
            inshape = input_shape
        return inshape[:-1]+(1,), inshape

    def compute_mask(self, inputs, mask=None):
        if self.extention:
            return mask[:, 1:], mask[:, 1:]
        return mask, mask

    def get_config(self):
        config = super(MilAttn, self).get_config()
        config.update(self._config_dict)
        return config

    def call(self, inputs, mask=None, training=None):
        assert mask is not None

        if self.extention:
            proc_inputs = inputs[:, 1:]
            # using implicit broadcast, alternatively tf.broadcast_to(concat, tf.shape(proc_inputs))
            concat = inputs[:, :1]
            zeros = tf.zeros_like(proc_inputs)
            _a = tf.math.abs(concat - proc_inputs)
            _b = concat * proc_inputs
            _c = concat + zeros
            attn_inputs = tf.concat([proc_inputs,_a,_b,_c], axis=-1)
        else:
            proc_inputs = inputs
            attn_inputs = inputs

        # V * h^T
        x = tf.math.tanh(tf.einsum('...ik, kj->...ij', attn_inputs, self.V))
        # (b, ts, h)
        if self.gated_attn:
            gate_x = tf.math.sigmoid(tf.einsum('...ik, kj->...ij', attn_inputs, self.U))
            att_x = x * gate_x
        else:
            att_x = x
        # w^T * [tanh(V * h^T) [+ gate]]
        logit_x = tf.einsum('...ik, kj->...ij', att_x, self.w)  # (b, ts, 1)
        # mask padded ts
        # _neg_inf = tf.math.minimum(tf.math.reduce_min(logit_x), 0.) + _LARGE_NEG(logit_x.dtype)
        # mask_adder = tf.expand_dims(1. - tf.cast(mask, dtype=logit_x.dtype), axis=-1) * _neg_inf
        mask_adder = tf.expand_dims(1. - tf.cast(mask, dtype=logit_x.dtype), axis=-1) * _LARGE_NEG(logit_x.dtype)
        logit = logit_x + mask_adder

        attn_scores = tf.math.softmax(logit, axis=-2)
        weighted_embedvecs = attn_scores * proc_inputs
        atts = tf.squeeze(attn_scores, axis=-1)
        return atts, weighted_embedvecs


class HWChannelSE(tkl.Layer):
    def __init__(self, unit, num_heads=1, use_bias=True, use_squeeze=False, squeeze_method=None,
                 gate_activate=None, gate_unit=0, **kwargs):  # epsilon=1e-6,
        super(HWChannelSE, self).__init__(**kwargs)
        self.supports_masking = True
        self.unit = unit
        self.num_heads = num_heads
        self.gate_activate = gate_activate
        self.gate_unit = gate_unit
        # self.eps = epsilon
        self.use_bias = use_bias
        self.use_squeeze = use_squeeze
        self.squeeze_method = squeeze_method
        if self.use_squeeze:
            assert self.squeeze_method in ['mean', 'var']
        else:
            assert self.squeeze_method is None
        self._config_dict = {
            'unit': unit,
            'use_bias': use_bias,
            'use_squeeze': use_squeeze,
            'squeeze_method': squeeze_method,
            # 'epsilon': epsilon,
            'num_heads': num_heads,
            'gate_unit': gate_unit,
            'gate_activate': gate_activate
        }

    def build(self, input_shape):
        input_depth = input_shape[-1]
        assert (self.unit % self.num_heads == 0)
        output_depth = self.unit // self.num_heads
        weight_init = 'glorot_uniform'

        if self.gate_activate and self.gate_unit>0:
            gatew_shape = (input_depth, self.gate_unit)
            gateb_shape = (self.gate_unit,)
            paramw_shape = (self.gate_unit, self.num_heads, output_depth)
            paramb_shape = (self.num_heads, output_depth)

            self.gate_weights = self.add_weight(name='interim_weight',
                                                shape=gatew_shape,
                                                initializer=weight_init
                                                )
            self.kernel = self.add_weight(name='sigmoid_weight',
                                          shape=paramw_shape,
                                          initializer='zeros'
                                          )
            if self.use_bias:
                self.gate_bias = self.add_weight(name='interim_bias',
                                                 shape=gateb_shape,
                                                 initializer='zeros'
                                                 )
                self.bias = self.add_weight(name='sigmoid_bias',
                                            shape=paramb_shape,
                                            initializer='zeros'
                                            )
            else:
                self.gate_bias = tf.zeros(shape=gateb_shape, dtype=TF_DTYPE)
                self.bias = tf.zeros(shape=paramb_shape, dtype=TF_DTYPE)
        else:
            self.gate_weights = None
            self.gate_bias = None
            paramw_shape = (input_depth, self.num_heads, output_depth)
            paramb_shape = (self.num_heads, output_depth)
            self.kernel = self.add_weight(name='sigmoid_weight',
                                          shape=paramw_shape,
                                          initializer=weight_init
                                          )
            if self.use_bias:
                self.bias = self.add_weight(name='sigmoid_bias',
                                            shape=paramb_shape,
                                            initializer='zeros'
                                            )
            else:
                self.bias = tf.zeros(shape=paramb_shape, dtype=TF_DTYPE)
        super(HWChannelSE, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return tf.math.reduce_any(mask, axis=-1)

    def call(self, inputs, mask=None):
        # tf.debugging.assert_all_finite(inputs, message=self.squeeze_method + 'SE: inputs check finite')
        indtype = inputs.dtype
        if mask is not None:
            mask_weights = tf.expand_dims(tf.cast(mask, dtype=indtype), axis=-1) + tf.zeros_like(inputs)
        else:
            mask_weights = tf.ones_like(inputs)
        mask_weights = tf.ensure_shape(mask_weights, inputs.shape)
        # squeeze
        if self.use_squeeze:
            squeezed = self.squeeze_step(inputs=inputs, mask_weights=mask_weights)
        else:
            squeezed = inputs
        # multi-head excite
        output = self.excite_step(squeezed)
        # concatenate heads
        output_shape = tf.concat([tf.shape(squeezed)[:-1], [self.unit, ]], axis=-1)
        return tf.reshape(output, shape=output_shape)

    def get_config(self):
        config = super(HWChannelSE, self).get_config()
        config.update(self._config_dict)
        return config

    def squeeze_step(self, inputs, mask_weights):
        mean, var = _masked_moments(inputs, mask_identity=mask_weights, axis=-2, keep_dims=True)
        if self.squeeze_method == 'var':
            return tf.math.subtract(var, 1.)
            # return tf.math.log(tf.math.add(var, 1e-3))
            # return tf.math.subtract(tf.math.sqrt(tf.math.add(var, 1e-6)), 1.)
        return mean

    def excite_step(self, sqz):
        if self.gate_activate and self.gate_unit > 0:
            _xw = tf.matmul(sqz, self.gate_weights)  # (.., gate_unit)
            if self.gate_activate == 'relu':
                gated = tf.nn.relu(tf.math.add(_xw, self.gate_bias))
            elif self.gate_activate == 'tanh':
                gated = tf.nn.tanh(tf.math.add(_xw, self.gate_bias))
            else:
                gated = tf.math.add(_xw, self.gate_bias)
            _gxw = tf.einsum('...c,chd->...hd', gated, self.kernel)
            # _gxw = tf.clip_by_value(_gxw, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)
            return tf.math.sigmoid(tf.math.add(_gxw, self.bias))
        else:
            _xw = tf.einsum('...c,chd->...hd', sqz, self.kernel)
            # _xw = tf.clip_by_value(_xw, clip_value_min=TF_DTYPE.min, clip_value_max=TF_DTYPE.max)
            return tf.math.sigmoid(tf.math.add(_xw, self.bias))


class MaskedMultiply(tkl.Layer):
    def __init__(self, residual_conn=True, **kwargs):
        super(MaskedMultiply, self).__init__(**kwargs)
        self.supports_masking = True
        self.residual_conn = residual_conn
        self._config_dict = {
            'residual_conn': residual_conn
        }

    def compute_mask(self, inputs, mask=None):
        inmask, _ = mask
        return inmask

    def call(self, inputs, mask=None, training=None):
        indata, chw = inputs
        if mask is not None:
            inmask, _ = mask
            if inmask is not None:
                mask_weights = tf.expand_dims(tf.cast(inmask, dtype=indata.dtype), axis=-1) + tf.zeros_like(indata)
            else:
                mask_weights = tf.ones_like(indata)
        else:
            mask_weights = tf.ones_like(indata)
        mask_weights.set_shape(indata.shape)

        output = tf.math.multiply(indata, chw)
        if self.residual_conn:
            output = tf.math.add(output, indata)
        return tf.math.multiply(mask_weights, output)

    def get_config(self):
        config = super(MaskedMultiply, self).get_config()
        config.update(self._config_dict)
        return config


def rFFBlock(_x, intermediate_size, dropout_rate=0., block_name='end_of_rFFB'):
    vec_size = _x.shape[-1]
    _y = tkl.Dense(intermediate_size, activation='relu')(_x)  # (batch_size, seq_len, dff)
    _y = tkl.Dense(vec_size)(_y)
    _y = tkl.Dropout(rate=dropout_rate)(_y)
    _z = tkl.Add()([_x, _y])
    return tkl.LayerNormalization(axis=-1, name=block_name)(_z)


# MAB(X, Y) = LayerNorm(H + rFF(H)), where H = LayerNorm(X + Multihead(X, Y, Y)),
def MABlock(_x, _y, num_heads, dropout_rate=0., block_name='end_of_MAB'):
    vector_size = _x.shape[-1]
    _key = MaskedMHLinear(unit=vector_size, num_heads=num_heads)(_y)
    _value = MaskedMHLinear(unit=vector_size, num_heads=num_heads)(_y)
    _query = MaskedMHLinear(unit=vector_size, num_heads=num_heads)(_x)
    atts, _resid = CrossAttnPooling(num_heads=num_heads,
                                    normalize=False,
                                    squeeze_dims=None,
                                    kqweights_center=False,
                                    kqweights_scale=False,
                                    name='resid_attn'
                                    )([_key, _value, _query])
    _r = GlobalSumPoolingRagged1D()(_resid)
    _r = MaskedMHLinear(unit=vector_size)(_r)
    _r = tkl.Dropout(rate=dropout_rate)(_r)
    _h = tkl.Add()([_x, _r])
    _h = tkl.LayerNormalization(axis=-1)(_h)
    return rFFBlock(_h, intermediate_size=vector_size*4, dropout_rate=dropout_rate, block_name=block_name)


def compile_bincrossentropy_model(model_inputs, model_outputs, model_name, is_logits=False, learn_rate=None):
    model = tk.Model(model_inputs, model_outputs, name=model_name)
    if not learn_rate:
        learn_rate = 1e-4
    loss_func = tk.losses.BinaryCrossentropy(from_logits=is_logits)
    metrics = [tk.metrics.BinaryAccuracy(), tk.metrics.AUC(), tk.metrics.Precision()]
    model.compile(optimizer=tk.optimizers.RMSprop(learn_rate),  # tk.optimizers.Adam(learn_rate),
                  loss=loss_func, metrics=metrics)
    return model


def build_embedding_inputs(inshape, conv_name, conv_pretrain, units=None, actv=None, normalize=False):
    input_qry = tkl.Input(shape=inshape[INPUT_KEYS['query']], dtype=TF_DTYPE, name=INPUT_KEYS['query'])
    input_tgt = tkl.Input(shape=inshape[INPUT_KEYS['target']], dtype=TF_DTYPE, name=INPUT_KEYS['target'])
    if all(inshape[INPUT_KEYS['query']]):
        input_query = tkl.Reshape(target_shape=(1,)+inshape[INPUT_KEYS['query']], name='reshape')(input_qry)
    else:
        input_query = tkl.Lambda(lambda _q: tf.expand_dims(_q, axis=1), name='reshape')(input_qry)
    combined_input = tkl.Concatenate(axis=-4, name='concat')([input_query, input_tgt])
    # TF2: masking inside TimeDistributed, at ConvEmbedLayer
    # combined_input = tkl.Masking(mask_value=0., name='masking')(combined_input)

    if units is not None and actv is not None:
        pooling = None
    else:
        pooling = 'avg'
    ce_layer = ConvEmbedLayer(
        image_shape=inshape[INPUT_KEYS['query']],
        pretrain=conv_pretrain,
        model_name=conv_name,
        pooling=pooling
    )
    TD = tkl.TimeDistributed(layer=ce_layer, name='TD')
    TD._always_use_reshape = True
    x = TD(combined_input)

    if units is not None and actv is not None:
        if units <= 0:
            out_units = x.shape[-1]
        else:
            out_units = units
        pce_layer = PostConvFlatten(output_units=out_units, output_actv=actv, l2norm=normalize)
        PTD = tkl.TimeDistributed(layer=pce_layer, name='PostTD')
        PTD._always_use_reshape = True
        x = PTD(x)

    return input_qry, input_tgt, x


def build_BERTembedding_inputs(inshape, bert_config, bert_pretrained, bert_trainable, bert_training_mode=None,
                               bert_pooler=None):
    input_claim = tkl.Input(shape=inshape[INPUT_KEYS['query']], dtype=tf.int32, name=INPUT_KEYS['query'])
    input_evids = tkl.Input(shape=inshape[INPUT_KEYS['target']], dtype=tf.int32, name=INPUT_KEYS['target'])
    if all(inshape[INPUT_KEYS['query']]):
        claim_input = tkl.Reshape((1,)+inshape[INPUT_KEYS['query']], name='reshape')(input_claim)
    else:
        claim_input = tkl.Lambda(lambda _z: tf.expand_dims(_z, axis=-2), name='reshape')(input_claim)
    combined_input = tkl.Concatenate(axis=-2, name='concat')([claim_input, input_evids])
    # TF2: masking inside TimeDistributed, at BERTEmbedLayer

    ce_layer = BERTEmbedLayer(
        config_file=bert_config,
        pretrained_file=bert_pretrained,
        bert_trainable=bert_trainable,
        training_mode=bert_training_mode,
        pooling=bert_pooler
    )
    TD = tkl.TimeDistributed(layer=ce_layer, name='TD')
    TD._always_use_reshape = True
    x = TD(combined_input)

    return input_claim, input_evids, x
