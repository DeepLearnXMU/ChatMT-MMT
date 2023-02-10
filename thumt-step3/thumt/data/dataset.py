# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math,os
import operator

import numpy as np
import tensorflow as tf
import thumt.utils.distribute as distribute


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs

def get_turn_position(file1):
    with open(file1, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    turn_position = []
    for line in content:
        tmp = []
        index = 0
        for i in line.split()[::-1]:
            #tmp.append(str(index))
            if i == '[SEP]':
                index += 1
            tmp.append(str(index))
        if len(line.split()) != len(tmp):
            print(line)
        turn_position.append(tmp)
    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.turn_position'
    if os.path.exists(position_file):
        return position_file
    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')
    #code.interact(local=locals())
    return position_file

def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        if distribute.is_distributed_training_mode():
            dataset = dataset.shard(distribute.size(), distribute.rank())

        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0)
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])

        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features

def get_training_input_contextual(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])
         
        dia_src_dataset = tf.data.TextLineDataset(params.train_dialog_en)
        dia_ctx_src_dataset = tf.data.TextLineDataset(params.train_dialog_ctx_en)
        dia_label_src_dataset = tf.data.TextLineDataset(params.train_dialog_en_label)
        sp_ctx_src_dataset = tf.data.TextLineDataset(params.train_speaker_ctx_en)
        sp_label_src_dataset = tf.data.TextLineDataset(params.train_speaker_en_label)

        position_file_dia_src = get_turn_position(params.train_dialog_ctx_en)
        position_file_sp_src = get_turn_position(params.train_speaker_ctx_en)
        position_dia_src_dataset = tf.data.TextLineDataset(position_file_dia_src)
        position_sp_src_dataset = tf.data.TextLineDataset(position_file_sp_src)
##########tgt
        dia_tgt_dataset = tf.data.TextLineDataset(params.train_dialog_de)
        dia_ctx_tgt_dataset = tf.data.TextLineDataset(params.train_dialog_ctx_de)
        dia_label_tgt_dataset = tf.data.TextLineDataset(params.train_dialog_de_label)
        sp_ctx_tgt_dataset = tf.data.TextLineDataset(params.train_speaker_ctx_de)
        sp_label_tgt_dataset = tf.data.TextLineDataset(params.train_speaker_de_label)

        position_file_dia_tgt = get_turn_position(params.train_dialog_ctx_de)
        position_file_sp_tgt = get_turn_position(params.train_speaker_ctx_de)
        position_dia_tgt_dataset = tf.data.TextLineDataset(position_file_dia_tgt)
        position_sp_tgt_dataset = tf.data.TextLineDataset(position_file_sp_tgt)
#        code.interact(local=locals())
        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, dia_src_dataset, dia_ctx_src_dataset, dia_label_src_dataset, sp_ctx_src_dataset, sp_label_src_dataset, position_dia_src_dataset, position_sp_tgt_dataset, dia_tgt_dataset, dia_ctx_tgt_dataset, dia_label_tgt_dataset, sp_ctx_tgt_dataset, sp_label_tgt_dataset, position_dia_tgt_dataset, position_sp_tgt_dataset))

        if distribute.is_distributed_training_mode():
            dataset = dataset.shard(distribute.size(), distribute.rank())

        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt, dia_src, dia_ctx_src, dia_label_src, sp_ctx_src, sp_label_src, pos_dia_src, pos_sp_src, dia_tgt, dia_ctx_tgt, dia_label_tgt, sp_ctx_tgt, sp_label_tgt, pos_dia_tgt, pos_sp_tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values,
                tf.string_split([dia_src]).values,
                tf.string_split([dia_ctx_src]).values,
                tf.string_split([dia_label_src]).values,
                tf.string_split([sp_ctx_src]).values,
                tf.string_split([sp_label_src]).values,
                tf.string_split([pos_dia_src]).values,
                tf.string_split([pos_sp_src]).values,
                tf.string_split([dia_tgt]).values,
                tf.string_split([dia_ctx_tgt]).values,
                tf.string_split([dia_label_tgt]).values,
                tf.string_split([sp_ctx_tgt]).values,
                tf.string_split([sp_label_tgt]).values,
                tf.string_split([pos_dia_tgt]).values,
                tf.string_split([pos_sp_tgt]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, dia_src, dia_ctx_src, dia_label_src, sp_ctx_src, sp_label_src, pos_dia_src, pos_sp_src, dia_tgt, dia_ctx_tgt, dia_label_tgt, sp_ctx_tgt, sp_label_tgt, pos_dia_tgt, pos_sp_tgt: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                dia_src,#tf.concat([src, [tf.constant(params.eos)], dia_src], axis=0),
                dia_ctx_src, 
                dia_label_src,
                sp_ctx_src,
                sp_label_src,
                pos_dia_src, 
                pos_sp_src,
                dia_tgt,#tf.concat([src, [tf.constant(params.eos)], dia_src], axis=0),
                dia_ctx_tgt,
                dia_label_tgt,
                sp_ctx_tgt,
                sp_label_tgt,
                pos_dia_tgt,
                pos_sp_tgt
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, dia_src, dia_ctx_src, dia_label_src, sp_ctx_src, sp_label_src, pos_dia_src, pos_sp_src, dia_tgt, dia_ctx_tgt, dia_label_tgt, sp_ctx_tgt, sp_label_tgt, pos_dia_tgt, pos_sp_tgt: {
                "source": src,
                "target": tgt,
                "dia_src": dia_src,
                "dia_ctx_src": dia_ctx_src,
                "dia_label_src": dia_label_src,
                "sp_ctx_src": sp_ctx_src,
                "sp_label_src": sp_label_src,
                "position_dia_src": pos_dia_src,
                "position_sp_src": pos_sp_src,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt),
                "dia_src_length": tf.shape(dia_src),
                "dia_ctx_src_length": tf.shape(dia_ctx_src),
                "sp_ctx_src_length": tf.shape(sp_ctx_src),
                "dia_tgt": dia_tgt,
                "dia_ctx_tgt": dia_ctx_tgt,
                "dia_label_tgt": dia_label_tgt,
                "sp_ctx_tgt": sp_ctx_tgt,
                "sp_label_tgt": sp_label_tgt,
                "position_dia_tgt": pos_dia_tgt,
                "position_sp_tgt": pos_sp_tgt,
                "dia_tgt_length": tf.shape(dia_tgt),
                "dia_ctx_tgt_length": tf.shape(dia_ctx_tgt),
                "sp_ctx_tgt_length": tf.shape(sp_ctx_tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        label_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["label"]),
            default_value=0
        )
        pos_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["position"]),
            default_value=-1
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["dia_src"] = src_table.lookup(features["dia_src"])
        features["dia_ctx_src"] = src_table.lookup(features["dia_ctx_src"])
        features["sp_ctx_src"] = src_table.lookup(features["sp_ctx_src"])
        features["sp_label_src"] = label_table.lookup(features["sp_label_src"])
        features["dia_label_src"] = label_table.lookup(features["dia_label_src"])
        features["position_dia_src"] = pos_table.lookup(features["position_dia_src"])
        features["position_sp_src"] = pos_table.lookup(features["position_sp_src"])

        features["dia_tgt"] = tgt_table.lookup(features["dia_tgt"])
        features["dia_ctx_tgt"] = tgt_table.lookup(features["dia_ctx_tgt"])
        features["sp_ctx_tgt"] = tgt_table.lookup(features["sp_ctx_tgt"])
        features["sp_label_tgt"] = label_table.lookup(features["sp_label_tgt"])
        features["dia_label_tgt"] = label_table.lookup(features["dia_label_tgt"])
        features["position_dia_tgt"] = pos_table.lookup(features["position_dia_tgt"])
        features["position_sp_tgt"] = pos_table.lookup(features["position_sp_tgt"])

        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])

        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])

        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        features["dia_src"] = tf.to_int32(features["dia_src"])
        features["dia_ctx_src"] = tf.to_int32(features["dia_ctx_src"])
        features["sp_ctx_src"] = tf.to_int32(features["sp_ctx_src"])

        features["dia_label_src"] = tf.to_int32(features["dia_label_src"])
        features["sp_label_src"] = tf.to_int32(features["sp_label_src"])

        features["position_dia_src"] = tf.to_int32(features["position_dia_src"])
        features["position_sp_src"] = tf.to_int32(features["position_sp_src"])

        features["dia_src_length"] = tf.to_int32(features["dia_src_length"])
        features["dia_src_length"] = tf.squeeze(features["dia_src_length"], 1)
        features["dia_ctx_src_length"] = tf.to_int32(features["dia_ctx_src_length"])
        features["dia_ctx_src_length"] = tf.squeeze(features["dia_ctx_src_length"], 1)

        features["sp_ctx_src_length"] = tf.to_int32(features["sp_ctx_src_length"])
        features["sp_ctx_src_length"] = tf.squeeze(features["sp_ctx_src_length"], 1)
#########tgt
        features["dia_tgt"] = tf.to_int32(features["dia_tgt"])
        features["dia_ctx_tgt"] = tf.to_int32(features["dia_ctx_tgt"])
        features["sp_ctx_tgt"] = tf.to_int32(features["sp_ctx_tgt"])

        features["dia_label_tgt"] = tf.to_int32(features["dia_label_tgt"])
        features["sp_label_tgt"] = tf.to_int32(features["sp_label_tgt"])

        features["position_dia_tgt"] = tf.to_int32(features["position_dia_tgt"])
        features["position_sp_tgt"] = tf.to_int32(features["position_sp_tgt"])

        features["dia_tgt_length"] = tf.to_int32(features["dia_tgt_length"])
        features["dia_tgt_length"] = tf.squeeze(features["dia_tgt_length"], 1)
        features["dia_ctx_tgt_length"] = tf.to_int32(features["dia_ctx_tgt_length"])
        features["dia_ctx_tgt_length"] = tf.squeeze(features["dia_ctx_tgt_length"], 1)

        features["sp_ctx_tgt_length"] = tf.to_int32(features["sp_ctx_tgt_length"])
        features["sp_ctx_tgt_length"] = tf.squeeze(features["sp_ctx_tgt_length"], 1)

        return features

def get_training_dialogue_input(filenames0, filenames1, filenames2, filenames3, filenames4, filenames5, params):

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames0)
        src_ctx_dataset = tf.data.TextLineDataset(filenames1)
        src_label = tf.data.TextLineDataset(filenames2)
        tgt_dataset = tf.data.TextLineDataset(filenames3)
        tgt_ctx_dataset = tf.data.TextLineDataset(filenames4)
        tgt_label = tf.data.TextLineDataset(filenames5)

        dataset = tf.data.Dataset.zip((src_dataset, src_ctx_dataset, src_label, tgt_dataset, tgt_ctx_dataset, tgt_label))

        if distribute.is_distributed_training_mode():
            dataset = dataset.shard(distribute.size(), distribute.rank())

        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, src_ctx, src_label, tgt, tgt_ctx, tgt_label: (
                tf.string_split([src]).values,
                tf.string_split([src_ctx]).values,
                tf.string_split([src_label]).values,
                tf.string_split([tgt]).values,
                tf.string_split([tgt_ctx]).values,
                tf.string_split([tgt_label]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, src_ctx, src_label, tgt, tgt_ctx, tgt_label: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([src_ctx, [tf.constant(params.eos)]], axis=0),
                src_label,
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt_ctx, [tf.constant(params.eos)]], axis=0),
                tgt_label
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, src_ctx, src_label, tgt, tgt_ctx, tgt_label: {
                "source": src,
                "source_ctx": src_ctx,
                "sp_label": src_label,
                "target": tgt,
                "target_ctx": tgt_ctx,
                "dia_label": tgt_label,
                "source_length": tf.shape(src),
                "source_ctx_length": tf.shape(src_ctx),
                "target_length": tf.shape(tgt),
                "target_ctx_length": tf.shape(tgt_ctx)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        dialog_features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        label_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["label"]),
            default_value=0
        )
        pos_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["position"]),
            default_value=-1
        )

        # Batching
        dialog_features = batch_examples(dialog_features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # String to index lookup
        dialog_features["source"] = src_table.lookup(dialog_features["source"])
        dialog_features["source_ctx"] = src_table.lookup(dialog_features["source_ctx"])
        dialog_features["target"] = src_table.lookup(dialog_features["target"])
        dialog_features["target_ctx"] = src_table.lookup(dialog_features["target_ctx"])

        dialog_features["source_length"] = tf.to_int32(dialog_features["source_length"])
        dialog_features["target_length"] = tf.to_int32(dialog_features["target_length"])
        dialog_features["source_ctx_length"] = tf.to_int32(dialog_features["source_ctx_length"])
        dialog_features["target_ctx_length"] = tf.to_int32(dialog_features["target_ctx_length"])

        dialog_features["source_length"] = tf.squeeze(dialog_features["source_length"], 1)
        dialog_features["target_length"] = tf.squeeze(dialog_features["target_length"], 1)
        dialog_features["source_ctx_length"] = tf.squeeze(dialog_features["source_ctx_length"], 1)
        dialog_features["target_ctx_length"] = tf.squeeze(dialog_features["target_ctx_length"], 1)

        dialog_features["sp_label"] = label_table.lookup(dialog_features["sp_label"])
        dialog_features["dia_label"] = label_table.lookup(dialog_features["dia_label"])

        dialog_features["sp_label"] = tf.to_int32(dialog_features["sp_label"])
        dialog_features["dia_label"] = tf.to_int32(dialog_features["dia_label"])

        return dialog_features

def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]

def get_evaluation_input_ctx(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []
        print(len(inputs))
        for i, data in enumerate(inputs):
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            if i > 0 and i < 2:
                dataset = dataset.map(
                    lambda x: x,
                    num_parallel_calls=params.num_threads
                )
            else:
                dataset = dataset.map(
                    lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                    num_parallel_calls=params.num_threads
                )

            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "context_dia_src": x[1],
                "position_dia_src": x[2],
                "context_dia_src_length": tf.shape(x[1])[0],
                "references": x[3:]
            },
            num_parallel_calls=params.num_threads
        )
#        code.interact(local=locals())
        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "context_dia_src": [tf.Dimension(None)],
                "context_dia_src_length": [],
                "position_dia_src":  [tf.Dimension(None)],
                "references": (tf.Dimension(None),) * (len(inputs) - 3)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "context_dia_src": params.pad,
                "context_dia_src_length": 0,
                "position_dia_src": params.pad,
                "references": (params.pad,) * (len(inputs) - 3)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        pos_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["position"]),
            default_value=-1
        )
        features["source"] = src_table.lookup(features["source"])
        features["context_dia_src"] = src_table.lookup(features["context_dia_src"])
        features["position_dia_src"] = pos_table.lookup(features["position_dia_src"])

    return features

def get_evaluation_input(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "references": x[1:]
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "references": (tf.Dimension(None),) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "references": (params.pad,) * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])

    return features


def get_inference_input(inputs, params):
    if params.generate_samples:
        batch_size = params.sample_batch_size
    else:
        batch_size = params.decode_batch_size

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )

        # Split string
        dataset = dataset.map(lambda x: tf.string_split([x]).values,
                              num_parallel_calls=params.num_threads)

        # Append <eos>
        dataset = dataset.map(
            lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=params.num_threads
        )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            batch_size * len(params.device_list),
            {"source": [tf.Dimension(None)], "source_length": []},
            {"source": params.pad, "source_length": 0}
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        features["source"] = src_table.lookup(features["source"])

        return features


def get_relevance_input(inputs, outputs, params):
    # inputs
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(inputs)
    )

    # Split string
    dataset = dataset.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset = dataset.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset = dataset.map(
        lambda x: {"source": x, "source_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(
        params.decode_batch_size,
        {"source": [tf.Dimension(None)], "source_length": []},
        {"source": params.pad, "source_length": 0}
    )

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["source"]),
        default_value=params.mapping["source"][params.unk]
    )
    features["source"] = src_table.lookup(features["source"])

    # outputs
    dataset_o = tf.data.Dataset.from_tensor_slices(
        tf.constant(outputs)
    )

    # Split string
    dataset_o = dataset_o.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset_o = dataset_o.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset_o = dataset_o.map(
        lambda x: {"target": x, "target_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset_o = dataset_o.padded_batch(
        params.decode_batch_size,
        {"target": [tf.Dimension(None)], "target_length": []},
        {"target": params.pad, "target_length": 0}
    )

    iterator = dataset_o.make_one_shot_iterator()
    features_o = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["target"]),
        default_value=params.mapping["target"][params.unk]
    )
    features["target"] = src_table.lookup(features_o["target"])
    features["target_length"] = features_o["target_length"]

    return features
