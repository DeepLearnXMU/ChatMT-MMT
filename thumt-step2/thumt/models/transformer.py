# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope("encoder", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
#    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
#                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope("decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias], reuse=tf.AUTO_REUSE):
#    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
#                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, dialog_features_en, dialog_features_de, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            src_embedding = tf.get_variable("weights",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            src_embedding = tf.get_variable("source_embedding",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    ### for dialogue features
    if mode == "train":
        ### for en-side
        sp_utterance = dialog_features_en["source"]
        sp_ctx = dialog_features_en["source_ctx"] # interlator or golden
        sp_len = dialog_features_en["source_length"]
        sp_ctx_len = dialog_features_en["source_ctx_length"]
        sp_mask = tf.sequence_mask(sp_len,
                                    maxlen=tf.shape(dialog_features_en["source"])[1],
                                    dtype=dtype or tf.float32)
        sp_ctx_mask = tf.sequence_mask(sp_ctx_len,
                                    maxlen=tf.shape(dialog_features_en["source_ctx"])[1],
                                    dtype=dtype or tf.float32)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            sp_bias = tf.get_variable("bias_sp", [hidden_size])
            sp_ctx_bias = tf.get_variable("bias_sp_ctx", [hidden_size])

        sp_inputs = tf.gather(src_embedding, sp_utterance)
        sp_ctx_inputs = tf.gather(src_embedding, sp_ctx)

        if params.multiply_embedding_mode == "sqrt_depth":
            sp_inputs = sp_inputs * (hidden_size ** 0.5)
            sp_ctx_inputs = sp_ctx_inputs * (hidden_size ** 0.5)

        sp_inputs = sp_inputs * tf.expand_dims(sp_mask, -1)
        sp_ctx_inputs = sp_ctx_inputs * tf.expand_dims(sp_ctx_mask, -1)

        sp_encoder_input = tf.nn.bias_add(sp_inputs, sp_bias)
        sp_ctx_encoder_input = tf.nn.bias_add(sp_ctx_inputs, sp_ctx_bias)
        sp_enc_attn_bias = layers.attention.attention_bias(sp_mask, "masking",
                                                        dtype=dtype)
        sp_ctx_enc_attn_bias = layers.attention.attention_bias(sp_ctx_mask, "masking",
                                                        dtype=dtype)
        if params.position_info_type == 'absolute':
            sp_encoder_input = layers.attention.add_timing_signal(sp_encoder_input)
            sp_ctx_encoder_input = layers.attention.add_timing_signal(sp_ctx_encoder_input)

        if params.residual_dropout:
            keep_prob = 1.0 - params.residual_dropout
            sp_encoder_input = tf.nn.dropout(sp_encoder_input, keep_prob)
            sp_ctx_encoder_input = tf.nn.dropout(sp_ctx_encoder_input, keep_prob)

        sp_encoder_output = transformer_encoder(sp_encoder_input, sp_enc_attn_bias, params)
        sp_ctx_encoder_output = transformer_encoder(sp_ctx_encoder_input, sp_ctx_enc_attn_bias, params)

        sp_mask = tf.expand_dims(sp_mask, -1)
        sp_rep = tf.reduce_sum(sp_encoder_output * sp_mask, -2) / tf.reduce_sum(sp_mask, -2)
        sp_ctx_rep = sp_ctx_encoder_output[:,0,:]
        fused_sp_rep = tf.concat([sp_rep, sp_ctx_rep], -1)

        dia_utterance = dialog_features_en["target"] # sample or golden
        dia_ctx = dialog_features_en["target_ctx"]
        dia_len = dialog_features_en["target_length"]
        dia_ctx_len = dialog_features_en["target_ctx_length"]
        dia_mask = tf.sequence_mask(dia_len,
                                    maxlen=tf.shape(dialog_features_en["target"])[1],
                                    dtype=dtype or tf.float32)
        dia_ctx_mask = tf.sequence_mask(dia_ctx_len,
                                    maxlen=tf.shape(dialog_features_en["target_ctx"])[1],
                                    dtype=dtype or tf.float32)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            dia_bias = tf.get_variable("bias_dia", [hidden_size])   
            dia_ctx_bias = tf.get_variable("bias_dia_ctx", [hidden_size])

        dia_inputs = tf.gather(src_embedding, dia_utterance)
        dia_ctx_inputs = tf.gather(src_embedding, dia_ctx)

        if params.multiply_embedding_mode == "sqrt_depth":
            dia_inputs = dia_inputs * (hidden_size ** 0.5)
            dia_ctx_inputs = dia_ctx_inputs * (hidden_size ** 0.5)

        dia_inputs = dia_inputs * tf.expand_dims(dia_mask, -1)
        dia_ctx_inputs = dia_ctx_inputs * tf.expand_dims(dia_ctx_mask, -1)

        dia_encoder_input = tf.nn.bias_add(dia_inputs, dia_bias)
        dia_ctx_encoder_input = tf.nn.bias_add(dia_ctx_inputs, dia_ctx_bias)
        dia_enc_attn_bias = layers.attention.attention_bias(dia_mask, "masking",
                                                        dtype=dtype)
        dia_ctx_enc_attn_bias = layers.attention.attention_bias(dia_ctx_mask, "masking",
                                                        dtype=dtype)
        if params.position_info_type == 'absolute':
            dia_encoder_input = layers.attention.add_timing_signal(dia_encoder_input)
            dia_ctx_encoder_input = layers.attention.add_timing_signal(dia_ctx_encoder_input)

        if params.residual_dropout:
            keep_prob = 1.0 - params.residual_dropout
            dia_encoder_input = tf.nn.dropout(dia_encoder_input, keep_prob)
            dia_ctx_encoder_input = tf.nn.dropout(dia_ctx_encoder_input, keep_prob)

        dia_encoder_output = transformer_encoder(dia_encoder_input, dia_enc_attn_bias, params)
        dia_ctx_encoder_output = transformer_encoder(dia_ctx_encoder_input, dia_ctx_enc_attn_bias, params)

        dia_mask = tf.expand_dims(dia_mask, -1)
        dia_rep = tf.reduce_sum(dia_encoder_output * dia_mask, -2) / tf.reduce_sum(dia_mask, -2)
        dia_ctx_rep = dia_ctx_encoder_output[:,0,:]
        fused_dia_rep = tf.concat([dia_rep, dia_ctx_rep], -1)

        ### for de-side
        sp_utterance = dialog_features_de["source"]
        sp_ctx = dialog_features_de["source_ctx"] # interlator or golden
        sp_len = dialog_features_de["source_length"]
        sp_ctx_len = dialog_features_de["source_ctx_length"]
        sp_mask = tf.sequence_mask(sp_len,
                                    maxlen=tf.shape(dialog_features_de["source"])[1],
                                    dtype=dtype or tf.float32)
        sp_ctx_mask = tf.sequence_mask(sp_ctx_len,
                                    maxlen=tf.shape(dialog_features_de["source_ctx"])[1],
                                    dtype=dtype or tf.float32)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            sp_bias = tf.get_variable("bias_sp", [hidden_size])
            sp_ctx_bias = tf.get_variable("bias_sp_ctx", [hidden_size])

        sp_inputs = tf.gather(src_embedding, sp_utterance)
        sp_ctx_inputs = tf.gather(src_embedding, sp_ctx)

        if params.multiply_embedding_mode == "sqrt_depth":
            sp_inputs = sp_inputs * (hidden_size ** 0.5)
            sp_ctx_inputs = sp_ctx_inputs * (hidden_size ** 0.5)

        sp_inputs = sp_inputs * tf.expand_dims(sp_mask, -1)
        sp_ctx_inputs = sp_ctx_inputs * tf.expand_dims(sp_ctx_mask, -1)

        sp_encoder_input = tf.nn.bias_add(sp_inputs, sp_bias)
        sp_ctx_encoder_input = tf.nn.bias_add(sp_ctx_inputs, sp_ctx_bias)
        sp_enc_attn_bias = layers.attention.attention_bias(sp_mask, "masking",
                                                        dtype=dtype)
        sp_ctx_enc_attn_bias = layers.attention.attention_bias(sp_ctx_mask, "masking",
                                                        dtype=dtype)
        if params.position_info_type == 'absolute':
            sp_encoder_input = layers.attention.add_timing_signal(sp_encoder_input)
            sp_ctx_encoder_input = layers.attention.add_timing_signal(sp_ctx_encoder_input)

        if params.residual_dropout:
            keep_prob = 1.0 - params.residual_dropout
            sp_encoder_input = tf.nn.dropout(sp_encoder_input, keep_prob)
            sp_ctx_encoder_input = tf.nn.dropout(sp_ctx_encoder_input, keep_prob)

        sp_encoder_output = transformer_encoder(sp_encoder_input, sp_enc_attn_bias, params)
        sp_ctx_encoder_output = transformer_encoder(sp_ctx_encoder_input, sp_ctx_enc_attn_bias, params)

        sp_mask = tf.expand_dims(sp_mask, -1)
        sp_rep = tf.reduce_sum(sp_encoder_output * sp_mask, -2) / tf.reduce_sum(sp_mask, -2)
        sp_ctx_rep = sp_ctx_encoder_output[:,0,:]
        fused_sp_rep_de = tf.concat([sp_rep, sp_ctx_rep], -1)

        dia_utterance = dialog_features_de["target"] # sample or golden
        dia_ctx = dialog_features_de["target_ctx"]
        dia_len = dialog_features_de["target_length"]
        dia_ctx_len = dialog_features_de["target_ctx_length"]
        dia_mask = tf.sequence_mask(dia_len,
                                    maxlen=tf.shape(dialog_features_de["target"])[1],
                                    dtype=dtype or tf.float32)
        dia_ctx_mask = tf.sequence_mask(dia_ctx_len,
                                    maxlen=tf.shape(dialog_features_de["target_ctx"])[1],
                                    dtype=dtype or tf.float32)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            dia_bias = tf.get_variable("bias_dia", [hidden_size])   
            dia_ctx_bias = tf.get_variable("bias_dia_ctx", [hidden_size])

        dia_inputs = tf.gather(src_embedding, dia_utterance)
        dia_ctx_inputs = tf.gather(src_embedding, dia_ctx)

        if params.multiply_embedding_mode == "sqrt_depth":
            dia_inputs = dia_inputs * (hidden_size ** 0.5)
            dia_ctx_inputs = dia_ctx_inputs * (hidden_size ** 0.5)

        dia_inputs = dia_inputs * tf.expand_dims(dia_mask, -1)
        dia_ctx_inputs = dia_ctx_inputs * tf.expand_dims(dia_ctx_mask, -1)

        dia_encoder_input = tf.nn.bias_add(dia_inputs, dia_bias)
        dia_ctx_encoder_input = tf.nn.bias_add(dia_ctx_inputs, dia_ctx_bias)
        dia_enc_attn_bias = layers.attention.attention_bias(dia_mask, "masking",
                                                        dtype=dtype)
        dia_ctx_enc_attn_bias = layers.attention.attention_bias(dia_ctx_mask, "masking",
                                                        dtype=dtype)
        if params.position_info_type == 'absolute':
            dia_encoder_input = layers.attention.add_timing_signal(dia_encoder_input)
            dia_ctx_encoder_input = layers.attention.add_timing_signal(dia_ctx_encoder_input)

        if params.residual_dropout:
            keep_prob = 1.0 - params.residual_dropout
            dia_encoder_input = tf.nn.dropout(dia_encoder_input, keep_prob)
            dia_ctx_encoder_input = tf.nn.dropout(dia_ctx_encoder_input, keep_prob)

        dia_encoder_output = transformer_encoder(dia_encoder_input, dia_enc_attn_bias, params)
        dia_ctx_encoder_output = transformer_encoder(dia_ctx_encoder_input, dia_ctx_enc_attn_bias, params)

        dia_mask = tf.expand_dims(dia_mask, -1)
        dia_rep = tf.reduce_sum(dia_encoder_output * dia_mask, -2) / tf.reduce_sum(dia_mask, -2)
        dia_ctx_rep = dia_ctx_encoder_output[:,0,:]
        fused_dia_rep_de = tf.concat([dia_rep, dia_ctx_rep], -1)


        return encoder_output, fused_sp_rep, fused_dia_rep, fused_sp_rep_de, fused_dia_rep_de

    return encoder_output


def decoding_graph(features, dialog_features_en, dialog_features_de, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    if params.position_info_type == 'absolute':
        decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]
    
    if mode == "train" and params.use_speaker:
        speaker_weight = tf.get_variable("speaker_weights",
                                        [2, hidden_size * 2],
                                        initializer=initializer, trainable=True)
        binary = tf.matmul(state["sp_rep"], speaker_weight, False, True)
        sp_ce_en = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary, labels=dialog_features_en["sp_labels"], smoothing=params.label_smoothing, normalize=True)

        binary = tf.matmul(state["sp_rep_de"], speaker_weight, False, True)
        sp_ce_de = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary, labels=dialog_features_de["sp_labels"], smoothing=params.label_smoothing, normalize=True)

        sp_loss = tf.reduce_mean(sp_ce_en) + tf.reduce_mean(sp_ce_de)
    if mode == "train" and params.use_dialogue:
        dialogue_weight = tf.get_variable("dialogue_weights",
                                        [2, hidden_size * 2],
                                        initializer=initializer, trainable=True)
        binary = tf.matmul(state["dia_rep"], dialogue_weight, False, True)
        dia_ce_en = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary, labels=dialog_features_en["dia_labels"], smoothing=params.label_smoothing, normalize=True)

        binary = tf.matmul(state["dia_rep_de"], dialogue_weight, False, True)
        dia_ce_de = losses.smoothed_softmax_cross_entropy_with_logits(logits=binary, labels=dialog_features_de["dia_labels"], smoothing=params.label_smoothing, normalize=True)
        dia_loss = tf.reduce_mean(dia_ce_en) + tf.reduce_mean(dia_ce_de)

    # label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss, sp_loss, dia_loss


def model_graph(features, dialog_features_en, dialog_features_de, mode, params):
    encoder_output = encoding_graph(features, dialog_features_en, dialog_features_de, mode, params)
    state = {
        "encoder": encoder_output
    }
    if mode == "train":
        encoder_output, sp_rep, dia_rep, sp_rep_de, dia_rep_de = encoding_graph(features, dialog_features_en, dialog_features_de, mode, params)
        state = {
            "encoder": encoder_output,
            "sp_rep": sp_rep,
            "dia_rep": dia_rep,
            "sp_rep_de": sp_rep_de,
            "dia_rep_de": dia_rep_de
        }
    output = decoding_graph(features, dialog_features_en, dialog_features_de, state, mode, params)

    return output


class Transformer(NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, dialog_features_en, dialog_features_de, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, dialog_features_en, dialog_features_de, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, features, features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.attention_key_channels or params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.attention_value_channels or params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, features, features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            use_speaker=False,
            use_dialogue=False,
            sp_alpha=0.0,
            dia_alpha=0.0,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.5,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # "absolute" or "relative"
            position_info_type="relative",
            # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            max_relative_dis=16
        )

        return params
