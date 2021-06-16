import tensorflow as tf
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import pickle
import gzip
import os
import json

path = '../data/squad'

max_seq = 256

path = '/hdd/user16/HT/data/squad/'

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


def serialize_example(feature0, feature1, feature2, feature3, feature4):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature_list(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _int64_feature(feature2),
        'feature3': _int64_feature(feature3),
        'feature4': _int64_feature(feature4),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_binary_file(file_name, lines, start_labels, end_labels, sep_positions, masks):
    print(f'Writing {file_name}.tfrecord...')
    with tf.io.TFRecordWriter(f'{file_name}.tfrecord') as writer:
        for i in tqdm(range(len(masks))):
            example = serialize_example(
                lines[i], start_labels[i], end_labels[i], sep_positions[i], masks[i])
            writer.write(example)
    print('Writing Complete!')


def write_pickle_file(file_name, lines, start_labels, end_labels, sep_positions, masks):
    lines = np.array(lines)
    start_labels = np.array(start_labels)
    end_labels = np.array(end_labels)
    sep_positions = np.array(sep_positions)
    masks = np.array(masks)

    print(f'Writing {file_name}.pickle...')
    with open(f'{file_name}.pickle', 'wb') as f:
        pickle.dump(lines, f)
        pickle.dump(start_labels, f)
        pickle.dump(end_labels, f)
        pickle.dump(sep_positions, f)
        pickle.dump(masks, f)
    print('Writing Complete!')


file_names = ['train-v1.1.json', 'dev-v1.1.json',
              'train-v2.0.json', 'dev-v2.0.json', ]

files = [path+x for x in file_names]


for file in files:
    #if os.path.exists(f'{file[:-5]}.tfrecord'): continue
    with open(file, 'r') as f:
        squad = json.load(f)
    lines = []

    start_label = []
    end_label = []

    sep_positions = []

    masks = []
    pads = [0 for _ in range(max_seq)]

    for i in tqdm(range(len(squad['data']))):
        
        for j in range(len(squad['data'][i]['paragraphs'])):
            context = squad['data'][i]['paragraphs'][j]['context']
            encoded = sp.EncodeAsIds(context)

            for k in range(len(squad['data'][i]['paragraphs'][j]['qas'])):
                if len(squad['data'][i]['paragraphs'][j]['qas'][k]['answers']) == 0: continue
                question = sp.EncodeAsIds(squad['data'][i]['paragraphs'][j]['qas'][k]['question'])

                if len(encoded) + len(question) <= max_seq - 2:
                    start =  squad['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['answer_start']
                    end = start + len(squad['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])
                    start_id = len(sp.EncodeAsIds(context[:start]))
                    end_id = len(sp.EncodeAsIds(context[:end])) - 1

                    tmp = [CLS] + question + [SEP] + encoded
            
                    start_label.append(len(question) + 2 + start_id)
                    end_label.append(len(question) + 2 + end_id)
                    masks.append(len(tmp))

                    tmp += pads[len(tmp):]
                    lines.append(tmp)

                    sep_position = tmp.index(SEP)
                    sep_positions.append(sep_position)
    
    print(len(lines), len(start_label), len(end_label), len(sep_positions), len(masks))
    
    if 'train' in file:
        write_binary_file(file[:-5], lines, start_label, end_label, sep_positions, masks)
    elif 'dev' in file:
        write_pickle_file(file[:-5], lines, start_label, end_label, sep_positions, masks)