import datetime
import tensorflow as tf
from bert import PretrainerBERT
from tqdm import tqdm

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer("./logs/" + current_time + '/train')

filenames = ['../BERT_data/xaa.pickle', 
             '../BERT_data/xab.pickle', 
             '../BERT_data/xac.pickle', 
             '../BERT_data/xad.pickle', 
             '../BERT_data/xae.pickle', 
             '../BERT_data/xaf.pickle',
             '../BERT_data/xag.pickle']


########################################### Preparing data ###########################################
dataset = tf.data.TFRecordDataset(filenames)

EPOCHS = 40
ACCUM_SIZE=300
BATCH_SIZE=128
BUFFER_SIZE = 50000000

hidden_size=128
dropout_rate=0.1
num_heads = 2
num_layers = 2
dff = 512
vocab_size = 30000
seq_len = 256

feature_description = {
    'nsp': tf.io.FixedLenFeature([seq_len], tf.int64),
    'nsp_label': tf.io.FixedLenFeature([], tf.int64),
    'sep': tf.io.FixedLenFeature([], tf.int64),
    'pad_mask': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example):
    return tf.io.parse_example(example, feature_description)

dataset = dataset.repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache().map(_parse_function)


########################################### Preparing model ###########################################
model = PretrainerBERT(num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)

nsp_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
mlm_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def mlm_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = mlm_loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-9)

def create_masks(inp):
    return



########################################### Training model ###########################################
max_steps = 1000000*ACCUM_SIZE
for step in range(max_steps):
    print(step)