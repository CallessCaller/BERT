import datetime
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
import numpy as np

from optimization import WarmUp, AdamWeightDecay
from bert import PretrainerBERTv2
from tqdm import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer("./logs_v2/" + current_time + '/train')

filenames = ['../BERT_data/xaa.tfrecord', 
             '../BERT_data/xab.tfrecord', 
             '../BERT_data/xac.tfrecord', 
             '../BERT_data/xad.tfrecord', 
             '../BERT_data/xae.tfrecord', 
             '../BERT_data/xaf.tfrecord',
             '../BERT_data/xag.tfrecord']


########################################### Preparing data ###########################################
dataset = tf.data.TFRecordDataset(filenames)

EPOCHS = 40
ACCUM_SIZE = 10
BATCH_SIZE = 100
BUFFER_SIZE = 5000000

hidden_size = 128
dropout_rate = 0.1
num_heads = 2
num_layers = 2
dff = 512
vocab_size = 30000
seq_len = 256
lr = 0.00176
total_step = 1000000

feature_description = {
    'feature0': tf.io.FixedLenFeature([seq_len], tf.int64),
    'feature1': tf.io.FixedLenFeature([], tf.int64),
    'feature2': tf.io.FixedLenFeature([], tf.int64),
    'feature3': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example):
    return tf.io.parse_example(example, feature_description)

dataset = dataset.repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(_parse_function)
iterator = iter(dataset)


########################################### Preparing model ###########################################
model = PretrainerBERTv2(num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)

nsp_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
mlm_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def mlm_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = mlm_loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-9)


@tf.function
def get_mlm_accuracy(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
    
@tf.function
def get_nsp_accuracy(real, pred):
    accuracies = tf.equal(real, tf.cast(tf.round(pred), dtype=tf.int64))
    accuracies = tf.cast(accuracies, tf.float32)
    
    return tf.reduce_mean(accuracies)
    

sequence_tensor = tf.convert_to_tensor(
    [[i for i in range(seq_len)] for _ in range(BATCH_SIZE)], 
    dtype=tf.int64)


def create_masks(tokens, sep, pad):
    sep = tf.reshape(sep, [BATCH_SIZE, 1])
    pad = tf.reshape(pad, [BATCH_SIZE, 1])
    
    sep_ids = tf.cast(tf.math.greater_equal(sequence_tensor, sep), dtype=tf.int64)
    pad_ids = tf.cast(tf.math.greater_equal(sequence_tensor, pad), dtype=tf.int64)

    tokens_np = tokens.numpy()
    p = np.random.uniform(0,1,1)

    mlm_position = np.random.randint(seq_len, size=38)
    label = np.zeros_like(tokens_np)
    label[:, mlm_position] = tokens_np[:, mlm_position]
    label = tf.convert_to_tensor(label, dtype=tf.int64)

    if p >= 0.9:
        tokens_np[:, mlm_position] = np.random.randint(vocab_size, size=1)
    elif p < 0.8:
        tokens_np[:, mlm_position] = 4
        position = np.zeros_like(tokens_np)
        position[:, mlm_position] = 1
        position = tf.convert_to_tensor(position, dtype=tf.int64)
        pad_ids = tf.maximum(pad_ids, position)

    return tokens_np, label, sep_ids, tf.cast(pad_ids, dtype=tf.float32)


lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=total_step, end_learning_rate=0.)
lr_schedule = WarmUp(initial_learning_rate=lr, decay_schedule_fn=lr_schedule, warmup_steps=10000)
#optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01)
optimizer = tfa.optimizers.LAMB(learning_rate=lr_schedule, weight_decay_rate=0.01)

checkpoint_path = "./checkpoints/train_v2"
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

########################################### Training model ###########################################
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

@tf.function
def build_model(input_ids, mlm_label, nsp_label, seg_ids, mask):
    
    with tf.GradientTape() as tape:
        mlm_prediction, nsp_prediction, pooled_output, _ = model(input_ids, seg_ids, mask)

    accum_gradients = [tf.zeros_like(x, dtype=tf.float32) for x in model.trainable_variables]

    return accum_gradients

@tf.function
def gradient_accumulation(input_ids, mlm_label, nsp_label, seg_ids, mask, accum_gradients):
    
    with tf.GradientTape() as tape:
        mlm_prediction, nsp_prediction, pooled_output, _ = model(input_ids, seg_ids, mask)
        mlm_loss = mlm_loss_function(mlm_label, mlm_prediction)
        nsp_loss = nsp_loss_object(nsp_label, nsp_prediction)

        loss = mlm_loss + nsp_loss
        loss /= ACCUM_SIZE

    gradients = tape.gradient(loss, model.trainable_variables)
    accum_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradients, gradients)]

    return loss, accum_gradients


@tf.function
def gradient_update(input_ids, mlm_label, nsp_label, seg_ids, mask, accum_gradients):
    
    with tf.GradientTape() as tape:
        mlm_prediction, nsp_prediction, pooled_output, _ = model(input_ids, seg_ids, mask)
        mlm_loss = mlm_loss_function(mlm_label, mlm_prediction)
        nsp_loss = nsp_loss_object(nsp_label, nsp_prediction)

        loss = mlm_loss + nsp_loss
        loss /= ACCUM_SIZE

    gradients = tape.gradient(loss, model.trainable_variables)
    accum_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradients, gradients)]
    optimizer.apply_gradients(zip(accum_gradients, model.trainable_variables))
    #optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    accum_gradients = [tf.zeros_like(x, dtype=tf.float32) for x in model.trainable_variables]

    mlm_acc = get_mlm_accuracy(mlm_label, mlm_prediction)
    nsp_acc = get_nsp_accuracy(nsp_label, nsp_prediction)
 

    return loss, accum_gradients, mlm_loss, nsp_loss, mlm_acc, nsp_acc, 
    


max_steps = total_step*ACCUM_SIZE
for step in tqdm(range(max_steps)):
    data = next(iterator)
    input_ids, mlm_label, seg_ids, mask = create_masks(data['feature0'], data['feature2'], data['feature3'])
    nsp_label = data['feature1']
        
    if step == 0:
       accum_gradients = build_model(input_ids, mlm_label, nsp_label, seg_ids, mask)
    elif (step + 1) % ACCUM_SIZE == 0:
        loss, accum_gradients, mlm_loss, nsp_loss, mlm_acc, nsp_acc = gradient_update(input_ids, mlm_label, nsp_label, seg_ids, mask, accum_gradients)
        
        if (step + 1) % 3 == 0:
            with writer.as_default():    
                tf.summary.scalar('Total Loss', loss*ACCUM_SIZE, step=(step//ACCUM_SIZE))
                tf.summary.scalar('MLM Loss', mlm_loss, step=(step//ACCUM_SIZE))
                tf.summary.scalar('NSP Loss', nsp_loss, step=(step//ACCUM_SIZE))
                tf.summary.scalar('MLM Accuracy', mlm_acc, step=(step//ACCUM_SIZE))
                tf.summary.scalar('NSP Accuracy', nsp_acc, step=(step//ACCUM_SIZE))
    else:
        loss, accum_gradients = gradient_accumulation(input_ids, mlm_label, nsp_label, seg_ids, mask, accum_gradients)

    if (step + 1) % (20000*ACCUM_SIZE) == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving Checkpoint for step {step+1} at {ckpt_save_path}...')

        