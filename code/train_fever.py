# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pickle
import math
import re
import multiprocessing as mp
import gc
from absl import app, flags

import tensorflow as tf
print(tf.version.VERSION)
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl

from transformers import AutoTokenizer
from modeling_cap import DELIMIT, INPUT_KEYS, LABEL_KEYS
from modeling_cap import load_oneline_tokens, build_BERTembedding_inputs, compile_bincrossentropy_model, create_virtual_gpus
from modeling_cap import  CrossAttnPooling, CrossDistAttnPooling, HWChannelSE, MSALayer, MilAttn, MABlock, \
    MaskedMHLayerNormalization, MaskedMHLinear, MaskedMultiply, SplitFirst, Similarities, SimilaritiesDual,\
    GlobalMaxPoolingRagged1D, GlobalSumPoolingRagged1D

num_workers = mp.cpu_count()
MAXIMUM_EVIDENCE = None  # 47  #
MAXIMUM_LEN = 96
sbert_pooler = 'cls'  # 'mean'  #
print('# of CPU =', num_workers, ', GPU =', tf.config.list_physical_devices('GPU'),  # avoid tf.test.is_gpu_available()
      ', tf =', tf.version.VERSION, ', CUDA =', tf.test.is_built_with_cuda())


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


def build_baseline(inshape, bert_pretrain, bert_config, bert_trainable, bert_trainmode=None,learn_rate=None,
                   attn_func=False, mcad=True, sce=False, num_heads=1):
    input_claim, input_evidence, z = build_BERTembedding_inputs(
        inshape=inshape,
        bert_pretrained=bert_pretrain,
        bert_config=bert_config,
        bert_trainable=bert_trainable,
        bert_training_mode=bert_trainmode,
        bert_pooler=sbert_pooler
    )
    embed_size = z.shape[-1]

    z = tkl.LayerNormalization(axis=-1, name='embed_ln')(z)
    _, trg_raw = SplitFirst(squeeze_first=False, name='split_first')(z)

    if num_heads > 1:
        z = MaskedMHLinear(unit=embed_size, num_heads=num_heads, name='embed_preattn_linear')(z)
    qry, trg = SplitFirst(squeeze_first=False, name='embed_split_first')(z)

    if sce:
        meanse = HWChannelSE(
            unit=embed_size,
            num_heads=num_heads,
            use_squeeze=True,
            use_bias=True,
            squeeze_method='mean',
            gate_unit=embed_size,
            gate_activate='relu',
            name='embed_meanse'
        )
        mutiplyse = MaskedMultiply(
            residual_conn=False,
            name='mse'
        )
        qry_meanse = meanse(qry)
        qry_mse = mutiplyse([qry, qry_meanse])
        trg_mse = mutiplyse([trg, qry_meanse])
    else:
        qry_mse = qry
        trg_mse = trg

    if attn_func:
        if mcad:
            print('MH-CAD')
            atts, x = CrossDistAttnPooling(num_heads=num_heads,
                                           squeeze_dims=[1,],
                                           kqweights=False,
                                           use_bias=True,
                                           name='mcad_pool'
                                           )([trg, trg_mse, qry])  # key, value, query
        else:
            print('MH-CA')
            sevar = HWChannelSE(
                unit=embed_size,
                num_heads=num_heads,
                use_squeeze=True,
                squeeze_method='var',
                gate_unit=embed_size,
                gate_activate='relu',
                use_bias=True,
                name='embed_varse'
            )(trg_raw)
            qry_varse = MaskedMultiply(residual_conn=False, name='qry_varse')([qry, sevar])

            atts, x = CrossAttnPooling(num_heads=num_heads,
                                       normalize=False,
                                       squeeze_dims=[1, ],
                                       kqweights_center=False,
                                       kqweights_scale=False,
                                       name='mca_pool')([trg, trg_mse, qry_varse])

        x = Similarities(has_weights=True, name='sim_pool')([qry_mse, x])
        y = GlobalSumPoolingRagged1D(keepdims=False, name='gsp')(x)
    else:
        y = Similarities(normalize=False,
                         has_weights=True,
                         name='sim_pool')([qry, trg])
        # y = GlobalAvgPoolingRagged1D(name='gap')(y)
        y = GlobalMaxPoolingRagged1D(name='gxp')(y)

    # # unlikely to work
    # y = GlobalPoolingRagged1D(pooling='max', name='gmp')(anc)
    # y = tkl.Dot(axes=-1, normalize=False, name='dot_product')([sig, y])
    # print(y) # (batch_size, 1)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(y) # (batch_size, 1)

    # y = tkl.Lambda(lambda _a: _a/(_a+1), name='sigm_log')(y)
    # output_main = tkl.Dense(1,
    #                         use_bias=False,
    #                         kernel_initializer=tf.initializers.ones(),
    #                         trainable=False,
    #                         name=LABEL_KEYS['main'])(y) # (batch_size, 1)

    cel_model = compile_bincrossentropy_model(model_inputs=[input_claim, input_evidence],
                                              model_outputs=[output_main],
                                              model_name='baseline',
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_pma(inshape, bert_pretrain, bert_config, bert_trainable, bert_trainmode=None,
              learn_rate=None, num_heads=1, num_layers=1, pma2=False):
    input_claim, input_evidence, z = build_BERTembedding_inputs(
        inshape=inshape,
        bert_pretrained=bert_pretrain,
        bert_config=bert_config,
        bert_trainable=bert_trainable,
        bert_training_mode=bert_trainmode,
        bert_pooler=sbert_pooler
    )
    embed_size = z.shape[-1]
    preln = tkl.LayerNormalization(axis=-1, name='embed_preattn_ln')
    z = preln(z)

    input_cls = tkl.Input(shape=(1,), dtype=tf.int32, name='cls')
    cls = tkl.Embedding(2, embed_size, input_length=1, mask_zero=True, trainable=True)(input_cls)
    cls = preln(cls)

    # rFF
    intermediate_size = embed_size * 4
    z = tkl.Dense(intermediate_size, activation='relu')(z)
    z = tkl.Dense(embed_size)(z)
    sig, anc = SplitFirst(squeeze_first=False, name='split_first')(z)
    if pma2:
        cls = tkl.Concatenate(axis=-2, name='pma2')([sig, cls])

    x = cls
    for i in range(num_layers):
        x = MABlock(x, anc, num_heads=num_heads, block_name='mab%d' % i)
    if pma2:
        sig, x = SplitFirst(squeeze_first=False, name='pma2split')(x)

    y = Similarities(has_weights=True, normalize=False, name='sim_pool')([sig, x])  # (batch_size, 1, 1)
    y = tkl.Reshape((1,), name='squeeze_k')(y)
    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(y)  # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_claim, input_evidence, input_cls],
                                              model_outputs=[output_main],
                                              model_name='milpma',
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_mca(inshape, bert_config, bert_pretrain, bert_trainable, bert_trainmode=None, learn_rate=None, mcad=False,
              num_heads=1, dropout_rate=None, se_interim_unit=None, num_layers=1, model_name=None,
              num_linproj=1, unweighted_probs=False, ma=False, aggr_ln=None, sce=True):
    if aggr_ln is None:
        aggr_ln = 'pre'
    assert aggr_ln in ['pre', 'post', 'noln']
    assert num_linproj in [0, 1, 2, 2.5, 3]
    if num_linproj == 0:
        num_heads = 1
    proj_init = None

    input_claim, input_evidence, z = build_BERTembedding_inputs(
        inshape=inshape,
        bert_pretrained=bert_pretrain,
        bert_config=bert_config,
        bert_trainable=bert_trainable,
        bert_training_mode=bert_trainmode,
        bert_pooler=sbert_pooler
    )
    # print(z.shape, num_heads)

    embed_size = z.shape[-1]
    z = tkl.LayerNormalization(axis=-1, name='embed_preattn_ln')(z)
    if dropout_rate != 0:
        _, anc_varse = SplitFirst(squeeze_first=False, name='pre-dropout_split_first')(z)  # MCA stdDev-excitation
        if dropout_rate is None:
            dropout_rate = max(embed_size / 256 * 0.1 - 0.3, 0)
        elif dropout_rate < 0:
            dropout_rate = max(embed_size / 256 * 0.05 - 0.2, 0)
        print('MCA/D dropout rate =', dropout_rate)
        dropout_shape = [None, 1, embed_size] if mcad else None
        z = tkl.Dropout(rate=dropout_rate,
                        noise_shape=dropout_shape,
                        name='embed_preattn_dropout')(z)

    sig_raw, anc_raw = SplitFirst(squeeze_first=False, name='split_first')(z)
    if dropout_rate == 0:
        anc_varse = anc_raw

    embed1 = embed2 = embed3 = None
    if num_linproj > 0:
        embed1 = MaskedMHLinear(
            unit=embed_size,
            num_heads=num_heads,
            kernel_initializer=proj_init,
            # post_norm=True, epsilon=1e-6, norm_axis=[-1], # , -2],
            # norm_adjust=True,
            name='embed_preattn_linear')
    if num_linproj > 1:
        embed2 = MaskedMHLinear(unit=embed_size, num_heads=num_heads, kernel_initializer=proj_init, name='embed2preattn_linear')
    if num_linproj > 2.5:
        embed3 = MaskedMHLinear(unit=embed_size, num_heads=num_heads, kernel_initializer=proj_init, name='embed3preattn_linear')

    if embed1 is not None:
        embed_tmp = embed1(z)
    else:
        embed_tmp = z
    sig_val, anc_val = SplitFirst(squeeze_first=False, name='embed_split_first')(embed_tmp)

    if embed2 is not None:
        assert embed1 is not None
        if embed3 is not None:
            sig_att = embed2(sig_raw)
            anc_att = embed3(anc_raw)
        elif num_linproj == 2:
            sig_att = embed2(sig_raw)
            anc_att = embed2(anc_raw)
        else:
            sig_att = embed2(sig_raw)
            anc_att = anc_val  # embed1(anc_tmp)
    else:
        assert embed3 is None
        sig_att, anc_att = sig_val, anc_val

    if sce:
        if se_interim_unit is None:
            se_interim_unit = embed_size // num_heads
        elif se_interim_unit < 0:
            se_interim_unit = embed_size
        meanse = HWChannelSE(
            unit=embed_size,
            num_heads=num_heads,
            use_squeeze=True,
            use_bias=True,
            squeeze_method='mean',
            gate_unit=se_interim_unit,
            gate_activate='relu',
            name='embed_meanse'
        )
        mutiplyse = MaskedMultiply(
            residual_conn=False,  # True if mcad else False
            name='mse'
        )
        sig_meanse = meanse(sig_raw)
        sig_mse = mutiplyse([sig_val, sig_meanse])
        anc_mse = mutiplyse([anc_val, sig_meanse])
        # anc_meanse = meanse(anc_raw)
        # anc_mse = mutiplyse([anc_val, anc_meanse])
    else:
        sig_mse = sig_val
        anc_mse = anc_val

    if aggr_ln == 'pre':
        postmse_norm = MaskedMHLayerNormalization(axis=-1,
                                                  num_heads=num_heads,
                                                  name='postmse_ln')
        vsig = postmse_norm([sig_mse, sig_mse])
        vanc = postmse_norm([anc_mse, anc_mse])
    else:
        vsig = sig_mse
        vanc = anc_mse

    if mcad:
        asig = sig_att
        aanc = anc_att
    else:
        sevar = HWChannelSE(
            unit=embed_size,
            num_heads=num_heads,
            use_squeeze=True,
            squeeze_method='var',
            gate_unit=se_interim_unit,
            gate_activate='relu',
            use_bias=True,
            name='embed_varse'
        )(anc_varse)
        sig_varse = MaskedMultiply(residual_conn=False, name='sig_varse')([sig_att, sevar])  # ([sig_mse, sestd])  #

        if ma:
            asig = sig_att  # ablation study
        else:
            asig = sig_varse
        aanc = anc_att

    if mcad:
        print('MH-CAD')
        atts, x = CrossDistAttnPooling(num_heads=num_heads,
                                       squeeze_dims=[1,],
                                       kqweights=False,  # True,  #
                                       use_bias=True,
                                       name='mcad_pool'
                                       )([aanc, vanc, asig])  # key, value, query
        # atts, x = CrossDistAttnPoolingSE(
        #     num_heads=num_heads,
        #     squeeze_dims=[1,],
        #     kqweights=False,  # True,  #
        #     name='mcad_pool'
        # )([aanc, vanc, asig, sedist])
    else:
        print('MH-CA')
        query = asig
        for i in range(num_layers):
            mca_name = 'mca_pool' if i == (num_layers-1) else 'mca_pool%d' % i
            atts, x = CrossAttnPooling(num_heads=num_heads,
                                       normalize=False,
                                       squeeze_dims=[1,],
                                       kqweights_center=False,
                                       kqweights_scale=False,
                                       name=mca_name)([aanc, vanc, query])  # key, value, query
            query = GlobalSumPoolingRagged1D(keepdims=True, name='gsp_%d' % i)(x)

    if aggr_ln == 'post':
        postaggr_ln = tkl.LayerNormalization(axis=-1, name='postaggr_ln')
        vsig = postaggr_ln(vsig)
        y = GlobalSumPoolingRagged1D(keepdims=True, name='gsp')(x)
        y = postaggr_ln(y)
        y = Similarities(has_weights=True, name='sim_pool')([vsig, y])
        y = tkl.Reshape(target_shape=(-1,), name='reduce_onedim')(y)
    else:
        if unweighted_probs:
            y, y_dual = SimilaritiesDual(has_weights=True, normalize=False, name='sim_pool')([vsig, x, vanc])
        else:
            y = Similarities(has_weights=True, normalize=False, name='sim_pool')([vsig, x])
        y = GlobalSumPoolingRagged1D(keepdims=False, name='gsp')(y)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(y)  # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_claim, input_evidence],
                                              model_outputs=[output_main],
                                              model_name=model_name,
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_model(inshape,
                bert_pretrain,
                bert_config,
                bert_trainable,
                bert_trainmode=None,
                numAttnLayers=12,
                num_heads=1,
                dropout_rate=0.,
                msa2=False,
                trfm_insize=None,
                trfm_inactv=None,
                trfm_outsize=None,
                trfm_outactv=None,
                learn_rate=None):
    input_claim, input_evidence, z = build_BERTembedding_inputs(
        inshape=inshape,
        bert_pretrained=bert_pretrain,
        bert_config=bert_config,
        bert_trainable=bert_trainable,
        bert_training_mode=bert_trainmode,
        bert_pooler=sbert_pooler
    )
    max_seq_len = 48 if MAXIMUM_EVIDENCE is None else (MAXIMUM_EVIDENCE + 1)
    vec_size = trfm_insize if trfm_insize is not None else z.shape[-1]
    preln = tkl.LayerNormalization(axis=-1, name='embed_preencoder_ln')
    z = preln(z)

    HIDDEN_MULTIPLE = 4
    attn_config = {
        'num_attention_heads': num_heads,
        'max_sequence_length': max_seq_len,
        'intermediate_size': int(HIDDEN_MULTIPLE * vec_size),
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': dropout_rate
    }
    transformer_encoder = MSALayer(
        transformer_config=attn_config,
        vec_size=vec_size,
        num_layers=numAttnLayers,
        return_final_encoder=False,
        return_all_encoder_outputs=False,
        trf_size=trfm_insize,
        trf_actv=trfm_inactv,
        pooler_size=trfm_outsize,
        pooler_actv=trfm_outactv,
        pre_layernorm=False,
        pooler_normalize=False,
        position_embeddings=False,
        name='transformer_encoder'
    )

    input_cls = tkl.Input(shape=(1,), dtype=tf.int32, name='cls')
    cls = tkl.Embedding(2, vec_size, input_length=1, mask_zero=True, trainable=True)(input_cls)
    # MSALayer NOT already has pre-encoder LN
    cls = preln(cls)

    sig, anc = SplitFirst(squeeze_first=False, name='split_first_pre')(z)
    x = tkl.Concatenate(axis=-2, name='concat_cls')([cls, anc])
    cls_encoder = transformer_encoder(x)
    if msa2:
        sig = transformer_encoder(z)

    y = Similarities(normalize=False, has_weights=True, name='sim_pool')([sig, cls_encoder])  # (b, 1, 1)
    y = tkl.Reshape((1,), name='squeeze_k')(y)
    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(y)  # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_claim, input_evidence, input_cls],
                                              model_outputs=[output_main],
                                              model_name='mil_transformer_encoder',
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_milattn(inshape, gated, bert_pretrain, bert_config, bert_trainable, bert_trainmode=None,
                  learn_rate=None, model_name=None):
    input_claim, input_evidence, z = build_BERTembedding_inputs(
        inshape=inshape,
        bert_pretrained=bert_pretrain,
        bert_config=bert_config,
        bert_trainable=bert_trainable,
        bert_training_mode=bert_trainmode,
        bert_pooler=sbert_pooler
    )
    z = tkl.LayerNormalization(axis=-1, name='embed_preattn_ln')(z)
    sig, anc = SplitFirst(squeeze_first=False, name='split_first')(z)

    # anc = SimilaritiesEW(
    #     # num_ew=4,
    #     name='sim_ew'
    # )([sig, anc])
    atts, y = MilAttn(hidden_dim=128, gated_attn=gated, name='attn_scores')(anc)

    x = Similarities(normalize=False, has_weights=True, name='sim_pool')([sig, y])
    # x = tkl.Dense(units=1, use_bias=False, name='sim_pool')(x)
    x = GlobalSumPoolingRagged1D(name='gsp')(x)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(x) # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_claim, input_evidence],
                                              model_outputs=[output_main],
                                              model_name=model_name,
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


# def split_data(file_name, randnum_gen, proportion=0.5):
#     num_lines = sum(1 for _ in open(file_name))
#     rand = randnum_gen.uniform(size=num_lines)
#     fst_list = []
#     snd_list = []
#     with open(file_name, 'r') as file:
#         linecount = 0
#         for line in file:
#             if rand[linecount]>proportion:
#                 fst_list.append(line)
#             else:
#                 snd_list.append(line)
#             linecount += 1
#     return fst_list, snd_list


flags.DEFINE_string('root_folder', 'FEVER/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('train', 'train_fever1.npy', 'Train dataset under the root directory')
flags.DEFINE_string('dev', 'dev_fever_all.npy', 'Dev dataset under the root directory')
# flags.DEFINE_string('test', 'test_fever_all.npy', 'Test dataset under the root directory')
flags.DEFINE_float('model_id', 2, 'one of the model_id = {0, 1, 2, 2.1, 3, 4}')
flags.DEFINE_string('ckpt_folder', 'ckpt/', 'Folder/path under root to store model checkpoint (.h5) files')
FLAGS = flags.FLAGS


def main(unused_argv):
    def pre0train(batch_size, input_shape, model_id, save_path, batch_per_epoch, distr_strategy, dev_ds,
                  early_stop=5, learnrate_schedule=None, validate_steps=None, pt_rmlist=None):
        print('pre0epoch training :', batch_size, '(', batch_per_epoch, ',', validate_steps, ')', early_stop)

        pds_train = load_pair(train, input_shape=input_shape, batch_size=batch_size, is_training=True,
                              cls=cls_placeholder, remove_list=pt_rmlist, shuffle_buffer=SHFBUF_SIZE)
        es_cb = tk.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stop,
            baseline=0.,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )
        pretrain_mode = False

        with distr_strategy.scope():
            if model_id == 0:
                p0t_model = build_baseline(inshape=input_shape,
                                           bert_config=bert_config_json,
                                           bert_pretrain=bert_pretrain_model,
                                           bert_trainable=False,
                                           bert_trainmode=pretrain_mode,
                                           learn_rate=learnrate_schedule
                                           )
            elif model_id == 1:
                p0t_model = build_milattn(inshape=input_shape,
                                          gated=attn_gate,
                                          bert_config=bert_config_json,
                                          bert_pretrain=bert_pretrain_model,
                                          bert_trainable=False,
                                          bert_trainmode=pretrain_mode,
                                          learn_rate=learnrate_schedule,
                                          model_name=attn_modelname
                                          )
            elif model_id == 2:
                p0t_model = build_mca(inshape=input_shape,
                                      mcad=mcadist,
                                      num_heads=nheads,
                                      num_linproj=num_mhlinproj,
                                      se_interim_unit=se_units,
                                      dropout_rate=pre0train_mca_dropout_rate,
                                      num_layers=1,
                                      bert_config=bert_config_json,
                                      bert_pretrain=bert_pretrain_model,
                                      bert_trainable=False,
                                      bert_trainmode=pretrain_mode,
                                      unweighted_probs=output_nonattn_probs,
                                      learn_rate=learnrate_schedule,
                                      model_name=mca_modelname
                                      )
            elif model_id == 3:
                p0t_model = build_pma(inshape=input_shape,
                                      num_heads=nheads,
                                      num_layers=1,
                                      pma2=pma_2,
                                      bert_config=bert_config_json,
                                      bert_pretrain=bert_pretrain_model,
                                      bert_trainable=False,
                                      bert_trainmode=pretrain_mode,
                                      learn_rate=learnrate_schedule
                                      )
            elif model_id == 4:
                p0t_model = build_model(inshape=input_shape,
                                        numAttnLayers=2,
                                        num_heads=nheads,
                                        # dropout_rate=pre0train_transformer_dropout_rate,
                                        msa2=msa_2,
                                        trfm_insize=None,
                                        trfm_inactv=None,
                                        trfm_outsize=None,
                                        trfm_outactv=None,
                                        bert_config=bert_config_json,
                                        bert_pretrain=bert_pretrain_model,
                                        bert_trainable=False,
                                        bert_trainmode=pretrain_mode,
                                        learn_rate=learnrate_schedule
                                        )
            # # ablation study
            # elif model_id == 5:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=False,
            #         sce=False,
            #         num_heads=1
            #     )
            # elif model_id == 6:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=True,
            #         sce=False,
            #         num_heads=1
            #     )
            # elif model_id == 7:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=False,
            #         sce=False,
            #         num_heads=nheads
            #     )
            # elif model_id == 8:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=True,
            #         sce=False,
            #         num_heads=nheads
            #     )
            # elif model_id == 9:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=False,
            #         sce=True,
            #         num_heads=nheads
            #     )
            # elif model_id == 10:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=True,
            #         sce=True,
            #         num_heads=nheads
            #     )
            # elif model_id == 11:
            #     p0t_model = build_mca(
            #         inshape=input_shape,
            #         mcad=False,
            #         ma=True,
            #         sce=False,
            #         aggr_ln='noln',
            #         num_heads=nheads,
            #         se_interim_unit=se_units,
            #         dropout_rate=pre0train_mca_dropout_rate,
            #         num_layers=1,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         unweighted_probs=output_nonattn_probs,
            #         learn_rate=learnrate_schedule
            #     )
            # elif model_id == 14:
            #     p0t_model = build_mca(
            #         inshape=input_shape,
            #         mcad=False,
            #         aggr_ln='post',
            #         num_heads=nheads,
            #         se_interim_unit=se_units,
            #         dropout_rate=pre0train_mca_dropout_rate,
            #         num_layers=1,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         unweighted_probs=output_nonattn_probs,
            #         learn_rate=learnrate_schedule
            #     )
            # elif model_id == 15:
            #     p0t_model = build_mca(
            #         inshape=input_shape,
            #         mcad=True,
            #         aggr_ln='post',
            #         num_heads=nheads,
            #         se_interim_unit=se_units,
            #         dropout_rate=pre0train_mca_dropout_rate,
            #         num_layers=1,
            #         bert_config=bert_config_json,
            #         bert_pretrain=bert_pretrain_model,
            #         bert_trainable=False,
            #         bert_trainmode=pretrain_mode,
            #         unweighted_probs=output_nonattn_probs,
            #         learn_rate=learnrate_schedule
            #     )
            else:
                raise NotImplementedError(f'Invalid Model ID: {model_id}')

        p0t_model.summary()
        p0t_model.fit(pds_train,
                      validation_data=dev_ds,
                      validation_steps=validate_steps,
                      epochs=num_epochs,
                      verbose=2,
                      callbacks=[es_cb]
                      )
        # Workaround TF BUG: .trainable WILL change the order of weights!
        td_layer = p0t_model.get_layer('TD')
        td_layer.trainable = True
        p0t_model.save(save_path, save_format='h5')
        # clean up
        sess = tf.compat.v1.keras.backend.get_session()
        tk.backend.clear_session()
        sess.close()
        del p0t_model, pds_train, es_cb
        print('pre0train gc =', gc.collect())
        return

    data_parent = FLAGS.root_folder
    os.chdir(data_parent)
    model_id = FLAGS.model_id
    train = FLAGS.train
    # test = FLAGS.test
    dev = FLAGS.dev
    cp_parent = FLAGS.ckpt_folder

    gpuram16gb = 15.8
    # Allocate GPU RAM
    batch_size_multiplier = 1.5  # limit 24GB, tf bug
    vgpu_ram_limit = int(1024 * gpuram16gb * batch_size_multiplier)
    logic_gpus = create_virtual_gpus(vgpu_ram=vgpu_ram_limit,
                                     # gpu_list=['0', '1'],
                                     onevgpu_per_physical=True,
                                     max_physicals=4
                                     )
    mgpu_strategy = tf.distribute.MirroredStrategy(
        devices=logic_gpus,
        cross_device_ops=tf.distribute.ReductionToOneDevice()
    )
    num_in_sync = mgpu_strategy.num_replicas_in_sync
    print('vGPU used:', len(logic_gpus), '==', num_in_sync, ', Logic GPUs =', logic_gpus)

    is_pre0trained = True
    line_shape = (None,)  # (MAXIMUM_LEN,)  #
    inshape = {INPUT_KEYS['query']: line_shape, INPUT_KEYS['target']: (MAXIMUM_EVIDENCE,) + line_shape}
    monitor_metric = 'val_binary_accuracy'
    min_metric = 0.
    early_stopping = 10
    num_epochs = 90
    train_samplesize, class_freq = npy_size(train, 2)  # (32857, [0.5008065, 0.4991935])  #
    val_size = 6616
    # test_size = 6613
    print(inshape, train, ': size =', train_samplesize, ';', dev, ': size =', val_size)

    bert_pretrain_model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    bert_config_json = 'base_model/'
    sbert_tokenizer = AutoTokenizer.from_pretrained(bert_pretrain_model, cache_dir=bert_config_json)
    if sbert_pooler == 'mean':
        rmlist = sbert_tokenizer.all_special_ids
    else:
        rmlist = None
    bert_train_mode = None

    nheads = 4
    # model_id = 0
    # model_id = 1
    attn_gate = True
    attn_modelname = 'milgattn' if attn_gate else 'milattn'
    # model_id = 2
    num_mhlinproj = 1
    se_units = -1
    output_nonattn_probs = False
    # model_id = 3
    pma_2 = True
    # model_id = 4
    msa_2 = True
    # dropout: model_id = 2,4
    pre0train_mca_dropout_rate = 0
    # pre0train_transformer_dropout_rate = 0.1
    mca_dropout_rate = pre0train_mca_dropout_rate
    # transformer_dropout_rate = pre0train_transformer_dropout_rate
    print('delim =', DELIMIT, ', pre-trained =', bert_pretrain_model, '(cache_folder =', bert_config_json, ')',
          sbert_pooler, rmlist, 'num_mh_linproj =', num_mhlinproj)

    headstr = str(nheads) if num_mhlinproj > 0 else 'none'
    mid_int = int(model_id)
    if mid_int == 2:
        mcadist = (model_id - mid_int) > 0
        model_id = mid_int
        str_hh = headstr + 'heads' + str(se_units) + 'se'
    else:
        mcadist = None
        if model_id > 2:
            str_hh = headstr + 'heads'
        else:
            str_hh = ''
    mca_modelname = 'milmcadse' if mcadist else 'milmcase'
    if output_nonattn_probs:
        mca_modelname += '_dualprobs'
    cls_placeholder = True if model_id in [3, 4] else False
    model_names = [
        'baseline', attn_modelname, mca_modelname, 'milpma', 'milmsa',
        # # ablation study
        # 'baseline_vema', 'baseline_dba', 'baseline_vemamh', 'baseline_dbamh', 'baseline_vemamhsce', 'baseline_dbamhsce',
        # 'milmase', 'milmcase_noln', 'milmcadse_noln', 'milmcase_postln', 'milmcadse_postln'
    ]
    print('**********************', model_names[model_id], '(model_id =', model_id, ';', str_hh, ')')

    BATCH_SIZE_PER_REPLICA = 12 if model_id > 1 else 16
    BATCH_SIZE_PER_REPLICA = int(BATCH_SIZE_PER_REPLICA * batch_size_multiplier)
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_in_sync
    SHFBUF_SIZE = -1  # 20000
    ds_train = load_pair(train, input_shape=inshape, batch_size=BATCH_SIZE, is_training=True,
                         shuffle_buffer=SHFBUF_SIZE, cls=cls_placeholder, remove_list=rmlist)
    ds_val = load_pair(dev, input_shape=inshape, batch_size=BATCH_SIZE, is_training=False, cls=cls_placeholder,
                       remove_list=rmlist)
    # ds_test = load_pair(test, input_shape=inshape, batch_size=BATCH_SIZE, is_training=False, cls=cls_placeholder,
    #                     remove_list=rmlist)
    num_batches_per_epoch = int(math.ceil(train_samplesize / BATCH_SIZE))
    validation_steps = int(math.ceil(1. * val_size / BATCH_SIZE))

    # continue training from ckpt, or from pre0trained, or from scratch
    if not os.path.exists(cp_parent):
        os.makedirs(cp_parent)
    ckpt_path = cp_parent + model_names[model_id] + str_hh + '_SBERT' + bert_pretrain_model.split('/')[-1] \
                + str(sbert_pooler) + 'pool'
    ckpt_file = ckpt_path + '_training' + str(bert_train_mode) + '_{epoch:03d}.hdf5'
    latest_ckpt = tf.train.latest_checkpoint(cp_parent)
    print('Loading latest checkpoint from', latest_ckpt)

    if latest_ckpt:
        initial_epoch = int(re.compile(r'_([0-9]+).').findall(latest_ckpt)[0])
    else:
        initial_epoch = 0
        if is_pre0trained:
            pre0trained = ckpt_path + '_pre0trained.hdf5'
            if not os.path.exists(pre0trained):
                frbsz = num_in_sync * (128 if model_id > 1 else 256)
                frbsz = int(frbsz * batch_size_multiplier)
                bpe = int(math.ceil(train_samplesize / frbsz))
                print('pre0epoch batch =', frbsz, bpe)
                if model_id != 2:
                    pt0lrb = [5 * bpe - 1, 10 * bpe - 1]
                    pt0lrv = [1e-4, 5e-5, 2e-5]
                else:
                    pt0lrb = [2 * bpe - 1, 5 * bpe - 1]
                    pt0lrv = [1e-4, 3e-5, 1e-5]
                pretrain_lr = tk.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=pt0lrb,
                    values=pt0lrv
                )
                pre0train(
                    distr_strategy=mgpu_strategy,
                    batch_size=frbsz,
                    input_shape=inshape,
                    model_id=model_id,
                    save_path=pre0trained,
                    batch_per_epoch=bpe,
                    early_stop=2,
                    learnrate_schedule=pretrain_lr,
                    validate_steps=validation_steps,
                    dev_ds=ds_val,
                    pt_rmlist=rmlist
                )
            latest_ckpt = pre0trained
            print('Loading pre0epoch-trained model from', latest_ckpt)

    if model_id < 2:
        lr_values = [2e-5, 1e-5]
        lr_boundaries = [5 * num_batches_per_epoch - 1]
    else:
        lr_values = [1e-5, 5e-6, 2e-6]
        lr_boundaries = [10 * num_batches_per_epoch - 1, 20 * num_batches_per_epoch - 1]
    lr_schedule = tk.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundaries,
        values=lr_values
    )
    print('init_epoch =', initial_epoch, lr_schedule, BATCH_SIZE, 'early_stop =', early_stopping,
          train, ':(', train_samplesize, class_freq, 'num_batches_per_epoch =', num_batches_per_epoch, ')')

    with mgpu_strategy.scope():
        if model_id == 0:
            train_model = build_baseline(inshape=inshape,
                                         bert_config=bert_config_json,
                                         bert_pretrain=bert_pretrain_model,
                                         bert_trainable=True,
                                         bert_trainmode=bert_train_mode,
                                         learn_rate=lr_schedule
                                         )
        elif model_id == 1:
            train_model = build_milattn(inshape=inshape,
                                        gated=attn_gate,
                                        bert_config=bert_config_json,
                                        bert_pretrain=bert_pretrain_model,
                                        bert_trainable=True,
                                        bert_trainmode=bert_train_mode,
                                        learn_rate=lr_schedule,
                                        model_name=attn_modelname
                                        )
        elif model_id == 2:
            train_model = build_mca(inshape=inshape,
                                    mcad=mcadist,
                                    num_heads=nheads,
                                    num_linproj=num_mhlinproj,
                                    se_interim_unit=se_units,
                                    dropout_rate=mca_dropout_rate,
                                    num_layers=1,
                                    bert_config=bert_config_json,
                                    bert_pretrain=bert_pretrain_model,
                                    bert_trainable=True,
                                    bert_trainmode=bert_train_mode,
                                    unweighted_probs=output_nonattn_probs,
                                    learn_rate=lr_schedule,
                                    model_name=mca_modelname
                                    )
        elif model_id == 3:
            train_model = build_pma(inshape=inshape,
                                    num_heads=nheads,
                                    num_layers=1,
                                    pma2=pma_2,
                                    bert_config=bert_config_json,
                                    bert_pretrain=bert_pretrain_model,
                                    bert_trainable=True,
                                    bert_trainmode=bert_train_mode,
                                    learn_rate=lr_schedule
                                    )
        elif model_id == 4:
            train_model = build_model(inshape=inshape,
                                      numAttnLayers=2,
                                      num_heads=nheads,
                                      # dropout_rate=transformer_dropout_rate,
                                      msa2=msa_2,
                                      trfm_insize=None,
                                      trfm_inactv=None,
                                      trfm_outsize=None,
                                      trfm_outactv=None,
                                      bert_config=bert_config_json,
                                      bert_pretrain=bert_pretrain_model,
                                      bert_trainable=True,
                                      bert_trainmode=bert_train_mode,
                                      learn_rate=lr_schedule
                                      )
        # # ablation study
        # elif model_id == 5:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=False,
        #         sce=False,
        #         num_heads=1
        #     )
        # elif model_id == 6:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=True,
        #         sce=False,
        #         num_heads=1
        #     )
        # elif model_id == 7:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=False,
        #         sce=False,
        #         num_heads=nheads
        #     )
        # elif model_id == 8:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=True,
        #         sce=False,
        #         num_heads=nheads
        #     )
        # elif model_id == 9:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=False,
        #         sce=True,
        #         num_heads=nheads
        #     )
        # elif model_id == 10:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=True,
        #         sce=True,
        #         num_heads=nheads
        #     )
        # elif model_id == 11:
        #     train_model = build_mca(
        #         inshape=inshape,
        #         mcad=False,
        #         ma=True,
        #         sce=False,
        #         aggr_ln='noln',
        #         num_heads=nheads,
        #         se_interim_unit=se_units,
        #         dropout_rate=mca_dropout_rate,
        #         num_layers=1,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         unweighted_probs=output_nonattn_probs,
        #         learn_rate=lr_schedule
        #     )
        # elif model_id == 14:
        #     train_model = build_mca(
        #         inshape=inshape,
        #         mcad=False,
        #         aggr_ln='post',
        #         num_heads=nheads,
        #         se_interim_unit=se_units,
        #         dropout_rate=mca_dropout_rate,
        #         num_layers=1,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         unweighted_probs=output_nonattn_probs,
        #         learn_rate=lr_schedule
        #     )
        # elif model_id == 15:
        #     train_model = build_mca(
        #         inshape=inshape,
        #         mcad=True,
        #         aggr_ln='post',
        #         num_heads=nheads,
        #         se_interim_unit=se_units,
        #         dropout_rate=mca_dropout_rate,
        #         num_layers=1,
        #         bert_config=bert_config_json,
        #         bert_pretrain=bert_pretrain_model,
        #         bert_trainable=True,
        #         bert_trainmode=bert_train_mode,
        #         unweighted_probs=output_nonattn_probs,
        #         learn_rate=lr_schedule
        #     )
        else:
            raise NotImplementedError(f'Invalid Model ID: {model_id}')
        if latest_ckpt is not None:
            print('Continue training, load whole model weights ..')
            train_model.load_weights(latest_ckpt, by_name=True)

    # ModelCheckpoint callback
    ccb = tk.callbacks.ModelCheckpoint(filepath=ckpt_file,
                                       save_weights_only=False,
                                       monitor=monitor_metric,
                                       mode='max',
                                       save_best_only=True,
                                       verbose=1
                                       )
    # ucb = CkptUpdateCallback(ckpt_fullpath=ckpt_file)
    # stopping callback
    scb = tk.callbacks.EarlyStopping(monitor=monitor_metric,
                                     patience=early_stopping,
                                     baseline=min_metric,
                                     verbose=1,
                                     mode='max'
                                     )
    # model fit
    train_model.summary()
    EPOCHS = initial_epoch + num_epochs
    try:
        train_model.fit(
            ds_train,
            validation_data=ds_val,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            verbose=2,
            callbacks=[ccb, scb]
        )
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(main)
