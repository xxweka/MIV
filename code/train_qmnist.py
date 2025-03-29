# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import math
import cv2
import re
import multiprocessing as mp
import gc
import pickle
from absl import app, flags

import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl

from modeling_cap import DELIMIT, DTYPE, TF_DTYPE, INPUT_KEYS, LABEL_KEYS, PRETRAIN_MODE
from modeling_cap import load_oneline, load_oneline_filenames, create_virtual_gpus,\
    build_embedding_inputs, compile_bincrossentropy_model
from modeling_cap import  CrossAttnPooling, CrossDistAttnPooling, HWChannelSE, MSALayer, MilAttn, MABlock,\
    MaskedMHLayerNormalization, MaskedMHLinear, MaskedMultiply, SplitFirst, Similarities, SimilaritiesDual,\
    GlobalMaxPoolingRagged1D, GlobalSumPoolingRagged1D

num_workers = mp.cpu_count()
print('# of CPU =', num_workers, ', tf =', tf.version.VERSION, tf.test.is_built_with_cuda(),
      tf.config.list_physical_devices('GPU'))
tk.backend.set_image_data_format('channels_last')
conv_pretrain_model = 'resnet18'
IMAGE_SIZE = (32, 32)  # w, h
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


def build_baseline(inshape, conv_pretrain, conv_name=None, learn_rate=None,
                   attn_func=False, mcad=None, sce=False, num_heads=1):
    input_qry, input_trg, z = build_embedding_inputs(inshape=inshape,
                                                     conv_pretrain=conv_pretrain,
                                                     # units=hidden_units,
                                                     # actv='relu',
                                                     conv_name=conv_name
                                                     )

    z = tkl.LayerNormalization(axis=-1, name='embed_ln')(z)
    _, trg_raw = SplitFirst(squeeze_first=False, name='split_first')(z)

    embed_size = z.shape[-1]
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
            residual_conn=False,  # True if mcad else False,
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
                name='embed_stdse'
            )(trg_raw)
            qry_varse = MaskedMultiply(residual_conn=False, name='sig_stdse')([qry, sevar])

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
        y = GlobalMaxPoolingRagged1D(name='gxp')(y)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(y)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_qry, input_trg],
                                              model_outputs=[output_main],
                                              model_name='baseline',
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_pma(inshape, conv_pretrain, conv_name=None, learn_rate=None,
              num_heads=1, num_layers=1, dropout_rate=0., pma2=True):
    input_signat, input_anchor, z = build_embedding_inputs(inshape=inshape,
                                                           conv_pretrain=conv_pretrain,
                                                           conv_name=conv_name
                                                           )
    embed_size = z.shape[-1]
    print('PMA_2 dropout rate =', dropout_rate)
    preln = tkl.LayerNormalization(axis=-1, name='embed_preattn_ln')

    z = preln(z)
    if dropout_rate != 0:
        z = tkl.Dropout(rate=dropout_rate)(z)
    input_cls = tkl.Input(shape=(1,), dtype=tf.int32, name='cls')
    cls = tkl.Embedding(2, embed_size, input_length=1, mask_zero=True, trainable=True)(input_cls)
    cls = preln(cls)
    cls = tkl.Dropout(rate=dropout_rate)(cls)

    # rFF
    intermediate_size = embed_size * 4
    z = tkl.Dense(intermediate_size, activation='relu')(z)
    z = tkl.Dense(embed_size)(z)
    sig, anc = SplitFirst(squeeze_first=False, name='split_first')(z)
    if pma2:
        cls = tkl.Concatenate(axis=-2, name='pma2')([sig, cls])

    y = cls
    for i in range(num_layers):
        y = MABlock(y, anc, num_heads=num_heads, dropout_rate=dropout_rate, block_name='mab%d' % i)
    if pma2:
        mi_query, mi_pooled = SplitFirst(squeeze_first=False, name='pma2split')(y)
    else:
        mi_query, mi_pooled = sig, y

    x = Similarities(has_weights=True, normalize=False, name='sim_pool')([mi_query, mi_pooled])  # (batch_size, 1, 1)
    x = tkl.Reshape((1,), name='squeeze_k')(x)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(x)  # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_signat, input_anchor, input_cls],
                                              model_outputs=[output_main],
                                              model_name='milpma',
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_mca(inshape, conv_pretrain, conv_name=None, learn_rate=None, mcad=False, ma=False, num_heads=1, num_layers=1,
              dropout_rate=None, dropout_align=True, se_interim_unit=None, model_name=None,
              num_linproj=1., unweighted_probs=False, aggr_ln=None, sce=True):
    if aggr_ln is None:
        aggr_ln = 'pre'
    assert aggr_ln in ['pre', 'post', 'noln']
    assert num_linproj in [0, 1, 2, 2.5, 3]
    if num_linproj == 0:
        num_heads = 1
    proj_init = None

    input_signat, input_anchor, z = build_embedding_inputs(inshape=inshape,
                                                           # units=-1,
                                                           # actv='relu',
                                                           conv_pretrain=conv_pretrain,
                                                           conv_name=conv_name
                                                           )
    # print(z.shape, z._keras_mask)
    embed_size = z.shape[-1]
    z = tkl.LayerNormalization(axis=-1, name='embed_preattn_ln')(z)

    if dropout_rate != 0:
        _, anc_varse = SplitFirst(squeeze_first=False, name='pre-dropout_split_first')(z)  # MCA Variance-excitation
        if dropout_rate is None:
            dropout_rate = max(embed_size / 256 * 0.1 - 0.3, 0)
        elif dropout_rate < 0:
            dropout_rate = max(embed_size / 256 * 0.05 - 0.2, 0)
        dropout_shape = [None, 1, embed_size] if dropout_align else None
        print('MCA/D dropout rate =', dropout_rate, ':', dropout_shape)
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
            embed2tmp = embed2(z)
            sig_att, anc_att = SplitFirst(squeeze_first=False, name='embed2split_first')(embed2tmp)
            # # equivalent
            # sig_att = embed2(sig_raw)
            # anc_att = embed2(anc_raw)
        else:
            sig_att = embed2(sig_raw)
            anc_att = anc_val  # embed1(anc_tmp)
    else:
        assert embed3 is None
        sig_att, anc_att = sig_val, anc_val

    if se_interim_unit is None:
        se_interim_unit = embed_size // num_heads
    elif se_interim_unit < 0:
        se_interim_unit = embed_size
    if sce:
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
            residual_conn=False,  # True if mcad else False,
            name='mse'
        )
        sig_meanse = meanse(sig_raw)
        sig_mse = mutiplyse([sig_val, sig_meanse])
        anc_mse = mutiplyse([anc_val, sig_meanse])
    else:
        sig_mse = sig_val
        anc_mse = anc_val

    if aggr_ln == 'pre':
        postmse_norm = MaskedMHLayerNormalization(axis=-1, num_heads=num_heads, name='postmse_ln')
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
            name='embed_stdse'
        )(anc_varse)
        sig_varse = MaskedMultiply(residual_conn=False, name='sig_stdse')([sig_att, sevar])  # ([sig_mse, sevar])  #

        if ma:
            asig = sig_att  # ablation study
        else:
            asig = sig_varse
        aanc = anc_att

    if mcad:
        # print('MH-CAD:', flush=True)
        atts, x = CrossDistAttnPooling(num_heads=num_heads,
                                       squeeze_dims=[1,],
                                       kqweights=False,  # True,  #
                                       use_bias=True,
                                       # l1_distance=False,
                                       name='mcad_pool'
                                       )([aanc, vanc, asig])  # key, value, query
    else:
        # print('MH-CA')
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
            y, y_dual = SimilaritiesDual(has_weights=True, name='sim_pool')([vsig, x, vanc])
        else:
            y = Similarities(has_weights=True, name='sim_pool')([vsig, x])
        y = GlobalSumPoolingRagged1D(keepdims=False, name='gsp')(y)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(y)  # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_signat, input_anchor],
                                              model_outputs=[output_main],
                                              model_name=model_name,
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_model(inshape,
                conv_pretrain,
                conv_name=None,
                numAttnLayers=12,
                num_heads=1,
                dropout_rate=0.,
                msa2=True,
                trfm_insize=None,
                trfm_inactv=None,
                trfm_outsize=None,
                trfm_outactv=None,
                learn_rate=None):
    input_signat, input_anchor, z = build_embedding_inputs(inshape=inshape,
                                                           conv_pretrain=conv_pretrain,
                                                           conv_name=conv_name
                                                           )
    vec_size = trfm_insize if trfm_insize is not None else z.shape[-1]
    preln = tkl.LayerNormalization(axis=-1, name='embed_preencoder_ln')
    z = preln(z)
    print('MSA_2 dropout rate =', dropout_rate)

    HIDDEN_MULTIPLE = 4
    attn_config = {
        'num_attention_heads': num_heads,
        'max_sequence_length': MAX_NUM_TARGETS + 1,
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
                            name=LABEL_KEYS['main'])(y) # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_signat, input_anchor, input_cls],
                                              model_outputs=[output_main],
                                              model_name='mil_transformer_encoder',
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


def build_milattn(inshape, gated, conv_pretrain, conv_name=None, learn_rate=None, model_name=None):
    input_signat, input_anchor, z = build_embedding_inputs(inshape=inshape,
                                                           # units=-1,
                                                           # actv='tanh',
                                                           conv_pretrain=conv_pretrain,
                                                           conv_name=conv_name
                                                           )
    z = tkl.LayerNormalization(axis=-1, name='embed_preattn_ln')(z)
    sig, anc = SplitFirst(squeeze_first=False, name='split_first')(z)
    atts, y = MilAttn(hidden_dim=128, gated_attn=gated, name='attn_scores')(anc)

    x = Similarities(normalize=False, has_weights=True, name='sim_pool')([sig, y])
    x = GlobalSumPoolingRagged1D(name='gsp')(x)

    output_main = tkl.Dense(1, activation='sigmoid',
                            use_bias=False,
                            kernel_initializer='ones',
                            trainable=False,
                            name=LABEL_KEYS['main'])(x) # (batch_size, 1)
    cel_model = compile_bincrossentropy_model(model_inputs=[input_signat, input_anchor],
                                              model_outputs=[output_main],
                                              model_name=model_name,
                                              is_logits=False,
                                              learn_rate=learn_rate
                                              )
    return cel_model


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


flags.DEFINE_string('root_folder', 'qmnist/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('train', 'train_qmnist1.tsv', 'Train dataset under the root directory')
flags.DEFINE_string('dev', 'dev_qmnist1.tsv', 'Dev dataset under the root directory')
# flags.DEFINE_string('test', 'test_qmnist1.tsv', 'Test dataset under the root directory')
flags.DEFINE_float('model_id', 2, 'one of the model_id = {0, 1, 2, 2.1, 3, 4}')
flags.DEFINE_string('ckpt_folder', 'ckpt/', 'Folder/path under root to store model files (.h5)')
FLAGS = flags.FLAGS


def main(unused_argv):
    def pre0train(batch_size, input_shape, model_id, save_path, batch_per_epoch, distr_strategy, dev_ds,
                  validate_steps=None, early_stop=2, learnrate_schedule=None):
        conv_pre0train_mode = PRETRAIN_MODE[0]
        print('pre0epoch:', batch_size, '(', batch_per_epoch, ',', validate_steps, ')', early_stop, SHFBUF_SIZE,
              conv_pre0train_mode)

        pds_train = load_pair_nocache(train, input_shape=input_shape, batch_size=batch_size, is_training=True,
                                      cls=cls_placeholder, shuffle_buffer=SHFBUF_SIZE, images_in_ram=data_cache)
        es_cb = tk.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stop,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        with distr_strategy.scope():
            if model_id == 0:
                p0t_model = build_baseline(inshape=input_shape,
                                           conv_pretrain=conv_pre0train_mode,
                                           conv_name=conv_pretrain_model,
                                           learn_rate=learnrate_schedule
                                           )
            elif model_id == 1:
                p0t_model = build_milattn(inshape=input_shape,
                                          gated=attn_gate,
                                          conv_pretrain=conv_pre0train_mode,
                                          conv_name=conv_pretrain_model,
                                          learn_rate=learnrate_schedule,
                                          model_name=attn_modelname
                                          )
            elif model_id == 2:
                p0t_model = build_mca(inshape=input_shape,
                                      mcad=mcadist,
                                      num_heads=nheads,
                                      num_linproj=num_mhlinproj,
                                      se_interim_unit=se_units,
                                      dropout_align=mca_dropout_align,
                                      dropout_rate=pre0train_mca_dropout_rate,
                                      conv_pretrain=conv_pre0train_mode,
                                      conv_name=conv_pretrain_model,
                                      aggr_ln='pre',
                                      unweighted_probs=output_nonattn_probs,
                                      learn_rate=learnrate_schedule,
                                      model_name=mca_modelname
                                      )
            elif model_id == 3:
                p0t_model = build_pma(inshape=input_shape,
                                      num_heads=nheads,
                                      num_layers=1,
                                      # dropout_rate=pre0train_transformer_dropout_rate,
                                      pma2=pma_2,
                                      conv_pretrain=conv_pre0train_mode,
                                      conv_name=conv_pretrain_model,
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
                                        conv_pretrain=conv_pre0train_mode,
                                        conv_name=conv_pretrain_model,
                                        learn_rate=learnrate_schedule
                                        )
            # # ablation study
            # elif model_id == 5:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         conv_pretrain=conv_pre0train_mode,
            #         conv_name=conv_pretrain_model,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=False,
            #         sce=False,
            #         num_heads=1
            #     )
            # elif model_id == 6:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         conv_pretrain=conv_pre0train_mode,
            #         conv_name=conv_pretrain_model,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=True,
            #         sce=False,
            #         num_heads=1
            #     )
            # elif model_id == 7:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         conv_pretrain=conv_pre0train_mode,
            #         conv_name=conv_pretrain_model,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=False,
            #         sce=False,
            #         num_heads=nheads
            #     )
            # elif model_id == 8:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         conv_pretrain=conv_pre0train_mode,
            #         conv_name=conv_pretrain_model,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=True,
            #         sce=False,
            #         num_heads=nheads
            #     )
            # elif model_id == 9:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         conv_pretrain=conv_pre0train_mode,
            #         conv_name=conv_pretrain_model,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=False,
            #         sce=True,
            #         num_heads=nheads
            #     )
            # elif model_id == 10:
            #     p0t_model = build_baseline(
            #         inshape=input_shape,
            #         conv_pretrain=conv_pre0train_mode,
            #         conv_name=conv_pretrain_model,
            #         learn_rate=learnrate_schedule,
            #         attn_func=True,
            #         mcad=True,
            #         sce=True,
            #         num_heads=nheads
            #     )
            # elif model_id == 11:
            #     p0t_model = build_mca(inshape=inshape,
            #                           mcad=False,
            #                           ma=True,
            #                           num_heads=nheads,
            #                           se_interim_unit=se_units,
            #                           dropout_rate=pre0train_mca_dropout_rate,
            #                           dropout_align=mca_dropout_align,
            #                           conv_pretrain=conv_pre0train_mode,
            #                           conv_name=conv_pretrain_model,
            #                           unweighted_probs=output_nonattn_probs,
            #                           sce=False,
            #                           aggr_ln='noln',  # 'pre',
            #                           learn_rate=learnrate_schedule
            #                           )
            # elif model_id == 12:
            #     p0t_model = build_mca(inshape=input_shape,
            #                           mcad=False,
            #                           num_heads=nheads,
            #                           se_interim_unit=se_units,
            #                           dropout_align=mca_dropout_align,
            #                           dropout_rate=pre0train_mca_dropout_rate,
            #                           conv_pretrain=conv_pre0train_mode,
            #                           conv_name=conv_pretrain_model,
            #                           aggr_ln='noln',
            #                           unweighted_probs=output_nonattn_probs,
            #                           learn_rate=learnrate_schedule
            #                           )
            # elif model_id == 13:
            #     p0t_model = build_mca(inshape=input_shape,
            #                           mcad=True,
            #                           num_heads=nheads,
            #                           se_interim_unit=se_units,
            #                           dropout_align=mca_dropout_align,
            #                           dropout_rate=pre0train_mca_dropout_rate,
            #                           conv_pretrain=conv_pre0train_mode,
            #                           conv_name=conv_pretrain_model,
            #                           aggr_ln='noln',
            #                           unweighted_probs=output_nonattn_probs,
            #                           learn_rate=learnrate_schedule
            #                           )
            # elif model_id == 14:
            #     p0t_model = build_mca(inshape=input_shape,
            #                           mcad=False,
            #                           num_heads=nheads,
            #                           se_interim_unit=se_units,
            #                           dropout_align=mca_dropout_align,
            #                           dropout_rate=pre0train_mca_dropout_rate,
            #                           conv_pretrain=conv_pre0train_mode,
            #                           conv_name=conv_pretrain_model,
            #                           aggr_ln='post',
            #                           unweighted_probs=output_nonattn_probs,
            #                           learn_rate=learnrate_schedule
            #                           )
            # elif model_id == 15:
            #     p0t_model = build_mca(inshape=input_shape,
            #                           mcad=True,
            #                           num_heads=nheads,
            #                           se_interim_unit=se_units,
            #                           dropout_align=mca_dropout_align,
            #                           dropout_rate=pre0train_mca_dropout_rate,
            #                           conv_pretrain=conv_pre0train_mode,
            #                           conv_name=conv_pretrain_model,
            #                           aggr_ln='post',
            #                           unweighted_probs=output_nonattn_probs,
            #                           learn_rate=learnrate_schedule
            #                           )
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

        # Workaround TF BUG: .trainable WILL change the order of weights (not by variable names)!
        td_layer = p0t_model.get_layer('TD')
        td_layer.trainable = True
        if base_mode == 1:
            for lyr in td_layer.layer.base_model.layers:
                if isinstance(lyr, tkl.BatchNormalization):
                    lyr.trainable = False  # TF2: BN.trainable=F automatically sets BN.training=F
                else:
                    lyr.trainable = True
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
    base_mode = 1  # exclBN # 3  # trainable,partBN #   4  #
    conv_pretrain = PRETRAIN_MODE[base_mode]
    query_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)  # h, w, c
    target_seq = MAX_NUM_TARGETS  # None  #
    inshape = {INPUT_KEYS['query']: query_shape, INPUT_KEYS['target']: (target_seq,) + query_shape}
    nheads = 2
    is_pre0trained = True if base_mode < 4 else False
    print(IMAGE_SIZE, 'delim =', DELIMIT, ', max_targets =', MAX_NUM_TARGETS, ', pre-trained =', conv_pretrain,
          conv_pretrain_model, ', num_heads =', nheads)

    model_id = FLAGS.model_id
    # model_id = 0
    # model_id = 1
    attn_gate = True
    attn_modelname = 'milgattn' if attn_gate else 'milattn'
    # model_id = 2
    num_mhlinproj = 1
    se_units = -1
    output_nonattn_probs = False
    mca_dropout_align = False
    pre0train_mca_dropout_rate = 0
    mca_dropout_rate = pre0train_mca_dropout_rate
    # model_id = 3
    pma_2 = True
    # model_id = 4
    msa_2 = True

    gpuram16gb = 15.8
    batch_size_multiplier = 1.5
    vgpu_ram_limit = int(1024 * gpuram16gb * batch_size_multiplier)
    logic_gpus = create_virtual_gpus(vgpu_ram=vgpu_ram_limit,
                                     gpu_list=['4', '5', '6', '7'],
                                     max_physicals=4,
                                     onevgpu_per_physical=True
                                     )
    mgpu_strategy = tf.distribute.MirroredStrategy(
        devices=logic_gpus,
        cross_device_ops=tf.distribute.ReductionToOneDevice()
    )
    num_in_sync = mgpu_strategy.num_replicas_in_sync
    print('vGPU used:', len(logic_gpus), '==', num_in_sync, '; Logic GPUs =', logic_gpus)

    min_early_stopping = 2
    monitor_metric = 'val_binary_accuracy'
    num_epochs = 180
    cache_filename = data_parent.split('/', 1)[0] + '.imgc'
    data_cache = build_images_cache(cache_filename, infile_list=['train60k/', 'test10k/'])
    print('save best by :', monitor_metric, ', # epochs =', num_epochs, ', data folder =', data_parent,
          'cached at', cache_filename, 'with size =', len(data_cache), 'num_mh_linproj =', num_mhlinproj)

    headstr = str(nheads) if num_mhlinproj > 0 else 'non'
    early_stopping = 30
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
    if model_id in [3, 4]:
        cls_placeholder = True
        early_stopping = 50
    else:
        cls_placeholder = False
        if model_id < 2 and base_mode == 1:
            early_stopping = 50
    model_names = [
        'baseline', attn_modelname, mca_modelname, 'milpma', 'milmsa',
        # # ablation study
        # 'baseline_vema', 'baseline_dba', 'baseline_vemamh', 'baseline_dbamh', 'baseline_vemamhsce', 'baseline_dbamhsce',
        # 'milmase', 'milmcase_noln', 'milmcadse_noln', 'milmcase_postln', 'milmcadse_postln'
    ]

    train = FLAGS.train
    dev = FLAGS.dev
    train_samplesize = sum(1 for _ in open(train))
    val_size = sum(1 for _ in open(dev))
    # test_size = sum(1 for _ in open(test))
    print('**********************', model_names[model_id], '(model_id =', model_id, ';', str_hh, ')')

    BATCH_SIZE_PER_REPLICA = 192 if model_id > 1 else 512
    BATCH_SIZE_PER_REPLICA = int(BATCH_SIZE_PER_REPLICA * batch_size_multiplier)
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_in_sync
    batches_per_epoch = int(math.ceil(1. * train_samplesize / BATCH_SIZE))
    validation_steps = int(math.ceil(1. * val_size / BATCH_SIZE))
    SHFBUF_SIZE = 0
    ds_train = load_pair_nocache(train, input_shape=inshape, batch_size=BATCH_SIZE, is_training=True,
                                 cls=cls_placeholder, shuffle_buffer=SHFBUF_SIZE, images_in_ram=data_cache)
    ds_val = load_pair_nocache(dev, input_shape=inshape, batch_size=BATCH_SIZE, is_training=False,
                               cls=cls_placeholder, images_in_ram=data_cache)
    # ds_test = load_pair_nocache(test, input_shape=inshape, batch_size=BATCH_SIZE, is_training=False,
    #                             cls=cls_placeholder, images_in_ram=data_cache)
    print('val_ds size =', val_size, '; train_ds size =', train_samplesize, 'from cache size =', len(data_cache),
          '; train/dev steps_per_epoch =', batches_per_epoch, validation_steps)

    # learning_rate
    if not is_pre0trained:
        learn_rates = [1e-4, 5e-5, 2e-5]
        cutoff = (30, 90) if model_id < 2 else (5, 20)
        boundaries = [cutoff[0] * batches_per_epoch, cutoff[1] * batches_per_epoch]
        lr_schedule = tk.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=learn_rates)
    else:
        cutoff = (20, 60) if model_id < 2 else (5, 20)
        boundaries = [cutoff[0] * batches_per_epoch, cutoff[1] * batches_per_epoch]
        lr_schedule = tk.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=[1e-4, 5e-5, 2e-5])

    # ckpt folder
    cp_parent = FLAGS.ckpt_folder
    if not os.path.exists(cp_parent):
        os.makedirs(cp_parent)
    ckpt_path = cp_parent + model_names[model_id] + str_hh + '_' + conv_pretrain_model + conv_pretrain
    latest_ckpt = tf.train.latest_checkpoint(cp_parent)
    print('Loading latest checkpoint from', latest_ckpt)

    # pre0epoch training
    if latest_ckpt:
        initial_epoch = int(re.compile(r'_([0-9]+).').findall(latest_ckpt)[0])
    else:
        initial_epoch = 0
        if is_pre0trained:
            pre0trained = ckpt_path + '_pre0trained.hdf5'
            if not os.path.exists(pre0trained):
                frbsz = num_in_sync * (192 if model_id > 1 else 512)  # (256 if model_id > 1 else 768)
                frbsz = int(frbsz * batch_size_multiplier)
                bpe = int(math.ceil(train_samplesize / frbsz))
                pre0train_lr = tk.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[5 * bpe - 1, 10 * bpe - 1],
                    values=[1e-4, 5e-5, 2e-5]
                )
                pre0train(
                    distr_strategy=mgpu_strategy,
                    batch_size=frbsz,
                    input_shape=inshape,
                    model_id=model_id,
                    save_path=pre0trained,
                    batch_per_epoch=bpe,
                    early_stop=min_early_stopping,
                    learnrate_schedule=pre0train_lr,
                    dev_ds=ds_val,
                    validate_steps=validation_steps
                )
            latest_ckpt = pre0trained
            print('Loading pre0epoch-trained model from', latest_ckpt)
    print(initial_epoch, lr_schedule, BATCH_SIZE, early_stopping, '; shuffle size =', SHFBUF_SIZE)

    # main training
    with mgpu_strategy.scope():
        if model_id == 0:
            train_model = build_baseline(inshape=inshape,
                                         conv_pretrain=conv_pretrain,
                                         conv_name=conv_pretrain_model,
                                         learn_rate=lr_schedule
                                         )
        elif model_id == 1:
            train_model = build_milattn(inshape=inshape,
                                        gated=attn_gate,
                                        conv_pretrain=conv_pretrain,
                                        conv_name=conv_pretrain_model,
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
                                    dropout_align=mca_dropout_align,
                                    conv_pretrain=conv_pretrain,
                                    conv_name=conv_pretrain_model,
                                    unweighted_probs=output_nonattn_probs,
                                    aggr_ln='pre',
                                    learn_rate=lr_schedule,
                                    model_name=mca_modelname
                                    )
        elif model_id == 3:
            train_model = build_pma(inshape=inshape,
                                    num_heads=nheads,
                                    num_layers=1,
                                    # dropout_rate=transformer_dropout_rate,
                                    pma2=pma_2,
                                    conv_pretrain=conv_pretrain,
                                    conv_name=conv_pretrain_model,
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
                                      conv_pretrain=conv_pretrain,
                                      conv_name=conv_pretrain_model,
                                      learn_rate=lr_schedule
                                      )
        # # ablation study
        # elif model_id == 5:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         conv_pretrain=conv_pretrain,
        #         conv_name=conv_pretrain_model,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=False,
        #         sce=False,
        #         num_heads=1
        #     )
        # elif model_id == 6:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         conv_pretrain=conv_pretrain,
        #         conv_name=conv_pretrain_model,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=True,
        #         sce=False,
        #         num_heads=1
        #     )
        # elif model_id == 7:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         conv_pretrain=conv_pretrain,
        #         conv_name=conv_pretrain_model,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=False,
        #         sce=False,
        #         num_heads=nheads
        #     )
        # elif model_id == 8:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         conv_pretrain=conv_pretrain,
        #         conv_name=conv_pretrain_model,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=True,
        #         sce=False,
        #         num_heads=nheads
        #     )
        # elif model_id == 9:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         conv_pretrain=conv_pretrain,
        #         conv_name=conv_pretrain_model,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=False,
        #         sce=True,
        #         num_heads=nheads
        #      )
        # elif model_id == 10:
        #     train_model = build_baseline(
        #         inshape=inshape,
        #         conv_pretrain=conv_pretrain,
        #         conv_name=conv_pretrain_model,
        #         learn_rate=lr_schedule,
        #         attn_func=True,
        #         mcad=True,
        #         sce=True,
        #         num_heads=nheads
        #     )
        # elif model_id == 11:
        #     train_model = build_mca(inshape=inshape,
        #                             mcad=False,
        #                             ma=True,
        #                             num_heads=nheads,
        #                             se_interim_unit=se_units,
        #                             dropout_rate=mca_dropout_rate,
        #                             dropout_align=mca_dropout_align,
        #                             conv_pretrain=conv_pretrain,
        #                             conv_name=conv_pretrain_model,
        #                             unweighted_probs=output_nonattn_probs,
        #                             sce=False,
        #                             aggr_ln='noln',  # 'pre',
        #                             learn_rate=lr_schedule
        #                             )
        # elif model_id == 12:
        #     train_model = build_mca(inshape=inshape,
        #                             mcad=False,
        #                             num_heads=nheads,
        #                             se_interim_unit=se_units,
        #                             dropout_rate=mca_dropout_rate,
        #                             dropout_align=mca_dropout_align,
        #                             conv_pretrain=conv_pretrain,
        #                             conv_name=conv_pretrain_model,
        #                             unweighted_probs=output_nonattn_probs,
        #                             aggr_ln='noln',
        #                             learn_rate=lr_schedule
        #                             )
        # elif model_id == 13:
        #     train_model = build_mca(inshape=inshape,
        #                             mcad=True,
        #                             num_heads=nheads,
        #                             se_interim_unit=se_units,
        #                             dropout_rate=mca_dropout_rate,
        #                             dropout_align=mca_dropout_align,
        #                             conv_pretrain=conv_pretrain,
        #                             conv_name=conv_pretrain_model,
        #                             unweighted_probs=output_nonattn_probs,
        #                             aggr_ln='noln',
        #                             learn_rate=lr_schedule
        #                             )
        # elif model_id == 14:
        #     train_model = build_mca(inshape=inshape,
        #                             mcad=False,
        #                             num_heads=nheads,
        #                             se_interim_unit=se_units,
        #                             dropout_rate=mca_dropout_rate,
        #                             dropout_align=mca_dropout_align,
        #                             conv_pretrain=conv_pretrain,
        #                             conv_name=conv_pretrain_model,
        #                             unweighted_probs=output_nonattn_probs,
        #                             aggr_ln='post',
        #                             learn_rate=lr_schedule
        #                             )
        # elif model_id == 15:
        #     train_model = build_mca(inshape=inshape,
        #                             mcad=True,
        #                             num_heads=nheads,
        #                             se_interim_unit=se_units,
        #                             dropout_rate=mca_dropout_rate,
        #                             dropout_align=mca_dropout_align,
        #                             conv_pretrain=conv_pretrain,
        #                             conv_name=conv_pretrain_model,
        #                             unweighted_probs=output_nonattn_probs,
        #                             aggr_ln='post',
        #                             learn_rate=lr_schedule
        #                             )
        else:
            raise NotImplementedError(f'Invalid Model ID: {model_id}')

        if latest_ckpt is not None:
            print('Continue training, load whole model weights ..')
            train_model.load_weights(latest_ckpt, by_name=True)

    # ModelCheckpoint callback
    # save_epochs = 1
    ckpt_file = ckpt_path + '_{epoch:03d}.hdf5'
    ccb = tk.callbacks.ModelCheckpoint(filepath=ckpt_file,
                                       save_weights_only=False,
                                       # save_freq=3,
                                       # period=save_epochs,
                                       monitor=monitor_metric,
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True
                                       )
    # ucb = CkptUpdateCallback(ckpt_fullpath=ckpt_file)
    # stopping callback
    min_metric = 0.
    scb = tk.callbacks.EarlyStopping(monitor=monitor_metric,
                                     patience=early_stopping,
                                     baseline=min_metric,
                                     verbose=1,
                                     mode='max'
                                     )

    train_model.summary()
    EPOCHS = initial_epoch + num_epochs
    try:
        train_model.fit(
            ds_train,
            validation_data=ds_val,
            validation_steps=validation_steps,  # MUST for mgpu_strategy
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            verbose=2,
            callbacks=[ccb, scb]
        )
    except Exception as e:
        print(e)

    # # clean up
    # sess = tf.compat.v1.keras.backend.get_session()
    # tk.backend.clear_session()
    # sess.close()
    # del train_model, vcb, ccb, scb, ucb, ds_train, ds_test, ds_val, test_x, test_y, y_val_enc, y_val, y_batch
    # del mgpu_strategy
    # print('*** End of Run # ' + run_str + ' for training', model_names[model_id], '(', model_id, ')', '; gc =',
    #       gc.collect())
    # mgpu_strategy = tf.distribute.MirroredStrategy(devices=logic_gpus)
    # num_in_sync = mgpu_strategy.num_replicas_in_sync

if __name__ == '__main__':
    app.run(main)
