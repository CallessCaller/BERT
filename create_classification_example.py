import tensorflow as tf
import numpy as np
import sentencepiece as spm
import os
from tqdm import tqdm
import pickle


path = '/hdd/user16/HT/data/GLUE'

spm_model = './30k-clean.model'
sp = spm.SentencePieceProcessor()
sp.load(spm_model)

max_seq = 256

pad = sp.piece_to_id('<pad>')
unk = sp.piece_to_id('<unk>')
CLS = sp.piece_to_id('[CLS]')
SEP = sp.piece_to_id('[SEP]')
MASK = sp.piece_to_id('[MASK]')

def _float_feature(values):
    """Returns a float_list from a float / double."""
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=[values]))
    return feature

def _int64_feature_list(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3, sts=False):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    if sts:
        feature = {
            'feature0': _int64_feature_list(feature0),
            'feature1': _float_feature(feature1),
            'feature2': _int64_feature(feature2),
            'feature3': _int64_feature(feature3),
        }
    else:
        feature = {
            'feature0': _int64_feature_list(feature0),
            'feature1': _int64_feature(feature1),
            'feature2': _int64_feature(feature2),
            'feature3': _int64_feature(feature3),
        }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_binary_file(file_name, lines, label, sep_positions, masks, sts=False):
    print(f'Writing {file_name}.tfrecord...')
    with tf.io.TFRecordWriter(f'{file_name}.tfrecord') as writer:
        for i in tqdm(range(len(masks))):
            example = serialize_example(
                lines[i], label[i], sep_positions[i], masks[i], sts)
            writer.write(example)
    print('Writing Complete!')

def write_pickle_file(file_name, lines, label, sep_positions, masks):
    lines = np.array(lines)
    label = np.array(label)
    sep_positions = np.array(sep_positions)
    masks = np.array(masks)

    print(f'Writing {file_name}.pickle...')
    with open(f'{file_name}.pickle', 'wb') as f:
        pickle.dump(lines, f)
        pickle.dump(label, f)
        pickle.dump(sep_positions, f)
        pickle.dump(masks, f)
    print('Writing Complete!')


file_names = ['/RTE/train.tsv', '/RTE/dev.tsv',
              '/CoLA/train.tsv', '/CoLA/dev.tsv',
              '/QNLI/train.tsv', '/QNLI/dev.tsv',
              '/SST-2/train.tsv', '/SST-2/dev.tsv',
              '/QQP/train.tsv', '/QQP/dev.tsv',
              '/MNLI/train.tsv', '/MNLI/dev_matched.tsv', '/MNLI/dev_mismatched.tsv',
              '/STS-B/train.tsv', '/STS-B/dev.tsv',
              '/MRPC/train.tsv', '/MRPC/dev.tsv',]

files = [path+x for x in file_names]

# read data
for idx, file in enumerate(files):
    lines = []
    label = []

    sep_positions = []
    sep_zeros = [0 for _ in range(max_seq)]
    sep_ones = [1 for _ in range(max_seq)]

    masks = []
    pads = [0 for _ in range(max_seq)]

    with open(file, 'r') as f:
        if 'RTE' in file or 'QNLI' in file:
            if not os.path.exists(f'{file[:-4]}.tfrecord'):
                for i, x in enumerate(f):
                    if i == 0: continue
                    line = x[:-1].split('\t')
                    tmp = [CLS] +  sp.EncodeAsIds(line[1]) + [SEP] + sp.EncodeAsIds(line[2])

                    if len(tmp) <= max_seq:
                        if 'not' in line[3]:
                            label.append(0)
                        else:
                            label.append(1)

                        masks.append(len(tmp))
                        
                        tmp = tmp + pads[len(tmp):]
                        lines.append(tmp)                 
                        
                        sep_position = tmp.index(SEP)                 
                        sep_positions.append(sep_position)
                
                if 'train' in file:
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)
                elif 'dev' in file:
                    write_pickle_file(file[:-4], lines, label, sep_positions, masks)
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)
        
        elif 'CoLA' in file or 'SST' in file:
            if not os.path.exists(f'{file[:-4]}.tfrecord'):
                for i, x in enumerate(f):
                    if i == 0: continue
                    line = x[:-1].split('\t')
                    if 'CoLA' in file:
                        tmp = [CLS] +  sp.EncodeAsIds(line[3]) + [SEP]
                    elif 'SST' in file:
                        tmp = [CLS] +  sp.EncodeAsIds(line[0]) + [SEP]

                    if len(tmp) <= max_seq:
                        label.append(int(line[1]))

                        masks.append(len(tmp))
                        
                        tmp = tmp + pads[len(tmp):]
                        lines.append(tmp)                 
                        
                        sep_position = tmp.index(SEP)                 
                        sep_positions.append(sep_position)

                if 'train' in file:
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)
                elif 'dev' in file:
                    write_pickle_file(file[:-4], lines, label, sep_positions, masks)
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)

        elif 'QQP' in file or 'MRPC' in file:
            if not os.path.exists(f'{file[:-4]}.tfrecord'):
                for i, x in enumerate(f):
                    if i == 0: continue
                    line = x[:-1].split('\t')
                    tmp = [CLS] +  sp.EncodeAsIds(line[3]) + [SEP] + sp.EncodeAsIds(line[4])

                    if len(tmp) <= max_seq:
                        if 'QQP' in file:
                            label.append(int(line[5]))
                        elif 'MRPC' in file:
                            label.append(int(line[0]))

                        masks.append(len(tmp))
                        
                        tmp = tmp + pads[len(tmp):]
                        lines.append(tmp)                 
                        
                        sep_position = tmp.index(SEP)                 
                        sep_positions.append(sep_position)

                if 'train' in file:
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)
                elif 'dev' in file:
                    write_pickle_file(file[:-4], lines, label, sep_positions, masks)
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)

        elif 'MNLI' in file:
            if not os.path.exists(f'{file[:-4]}.tfrecord'):
                for i, x in enumerate(f):
                    if i == 0: continue
                    line = x[:-1].split('\t')
                    tmp = [CLS] +  sp.EncodeAsIds(line[8]) + [SEP] + sp.EncodeAsIds(line[9])

                    if len(tmp) <= max_seq:
                        if 'contradiction' in line[-1]:
                            label.append(0)
                        elif 'neutral' in line[-1]:
                            label.append(1)
                        elif 'entailment' in line[-1]:
                            label.append(2)

                        masks.append(len(tmp))
                        
                        tmp = tmp + pads[len(tmp):]
                        lines.append(tmp)                 
                        
                        sep_position = tmp.index(SEP)                 
                        sep_positions.append(sep_position)

                if 'train' in file:
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)
                elif 'dev' in file:
                    write_pickle_file(file[:-4], lines, label, sep_positions, masks)
                    write_binary_file(file[:-4], lines, label, sep_positions, masks)

        elif 'STS' in file:
            if not os.path.exists(f'{file[:-4]}.tfrecord'):
                for i, x in enumerate(f):
                    if i == 0: continue
                    line = x[:-1].split('\t')
                    tmp = [CLS] +  sp.EncodeAsIds(line[7]) + [SEP] + sp.EncodeAsIds(line[8])

                    if len(tmp) <= max_seq:
                        label.append(float(line[-1]))

                        masks.append(len(tmp))
                        
                        tmp = tmp + pads[len(tmp):]
                        lines.append(tmp)                 
                        
                        sep_position = tmp.index(SEP)                 
                        sep_positions.append(sep_position)

                if 'train' in file:
                    write_binary_file(file[:-4], lines, label, sep_positions, masks, True)
                elif 'dev' in file:
                    write_pickle_file(file[:-4], lines, label, sep_positions, masks)
                    write_binary_file(file[:-4], lines, label, sep_positions, masks, True)


