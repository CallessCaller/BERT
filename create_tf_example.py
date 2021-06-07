import tensorflow as tf
import gzip
import pickle
import numpy as np
from tqdm import tqdm

def _float_feature(values):
    """Returns a float_list from a float / double."""
    feature = tf.train.Feature(int64_list=tf.train.FloatList(value=list(values)))
    return feature


def _int64_feature_list(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
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
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


filenames = ['../BERT_data/xab.pickle', 
             '../BERT_data/xac.pickle', 
             '../BERT_data/xad.pickle', 
             '../BERT_data/xae.pickle', 
             '../BERT_data/xaf.pickle',
             '../BERT_data/xag.pickle']


for filename in filenames: 
    print(f'Loading {filename}...')
    with open(f'/hdd/user16/HT/BERT_data/{filename}', 'rb') as f:
        nsp = pickle.load(f)
        nsp_label = pickle.load(f)
        seg = pickle.load(f)
        mask = pickle.load(f)
    print('Loading Complete!')
    
    print(f'Writing {filename[:-7]}.tfrecord...')
    with tf.io.TFRecordWriter(f'{filename[:-7]}.tfrecord') as writer:
        for i in tqdm(range(len(mask))):
            example = serialize_example(
                nsp[i], nsp_label[i], seg[i], mask[i])
            writer.write(example)
    print('Writing Complete!')