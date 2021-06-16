from bert import PretrainerBERT
from bert_classifier import SquadBERT
import sentencepiece as spm
import datetime
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
import numpy as np
import pickle
import json

from optimization import WarmUp, AdamWeightDecay
from tqdm import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision


spm_model = './30k-clean.model'
sp = spm.SentencePieceProcessor()
sp.load(spm_model)

path = '../data/squad/'
hidden_size = 128
dropout_rate = 0.1
num_heads = 2
num_layers = 2
dff = 512
vocab_size = 30000
seq_len = 256

num_data = {
    'v1.1':  58783,
    'v2.0':  58286,
}

feature_description = {
    'feature0': tf.io.FixedLenFeature([seq_len], tf.int64),
    'feature1': tf.io.FixedLenFeature([], tf.int64),
    'feature2': tf.io.FixedLenFeature([], tf.int64),
    'feature3': tf.io.FixedLenFeature([], tf.int64),
    'feature4': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example):
    return tf.io.parse_example(example, feature_description)


@tf.function
def get_accuracy(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=-1))
    accuracies = tf.cast(accuracies, tf.float32)
    
    return tf.reduce_mean(accuracies), tf.reduce_sum(accuracies)


def train(version, epochs, batch_size, warm_up, lr):
    best = 0
    bert = PretrainerBERT(num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)
    optimizer = tfa.optimizers.LAMB(learning_rate=0.00176, weight_decay_rate=0.01)
    checkpoint_path = "./checkpoints/train/ckpt-3"
    ckpt = tf.train.Checkpoint(model=bert, optimizer=optimizer)
    ckpt.read(checkpoint_path).expect_partial()

    print('Latest checkpoint restored!!')

    classifier = SquadBERT(num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"./{version}/logs/" + current_time + '/train')

    dataset = tf.data.TFRecordDataset(f'{path}train-{version}.tfrecord')
    
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    total_step = (num_data[version]*EPOCHS)//BATCH_SIZE
    BUFFER_SIZE = 50
    warm_up_steps = warm_up

    lr = lr

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=total_step, end_learning_rate=0.)
    lr_schedule = WarmUp(initial_learning_rate=lr, decay_schedule_fn=lr_schedule, warmup_steps=warm_up_steps)
    optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01)

    dataset = dataset.repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(_parse_function)

    with open(f'{path}dev-{version}.pickle', 'rb') as f:
        test_lines = pickle.load(f)
        test_start_labels = pickle.load(f)
        test_end_labels = pickle.load(f)
        test_sep = pickle.load(f)
        test_mask = pickle.load(f)
        ids = pickle.load(f)

    start_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    end_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    sequence_tensor = tf.convert_to_tensor(
        [[i for i in range(seq_len)] for _ in range(BATCH_SIZE)], 
        dtype=tf.int64)

    def create_masks(sep, pad):
        sep = tf.convert_to_tensor(sep)
        pad = tf.convert_to_tensor(pad)
        sep = tf.reshape(sep, [-1, 1])
        pad = tf.reshape(pad, [-1, 1])
        
        if sep.shape[0] == BATCH_SIZE:
            sep_ids = tf.cast(tf.math.greater_equal(sequence_tensor, sep), dtype=tf.int64)
            pad_ids = tf.cast(tf.math.greater_equal(sequence_tensor, pad), dtype=tf.int64)
        else:
            trim_sequence_tensor = tf.convert_to_tensor([[i for i in range(seq_len)] for _ in range(sep.shape[0])],
                                                    dtype=tf.int64)
            sep_ids = tf.cast(tf.math.greater_equal(trim_sequence_tensor, sep), dtype=tf.int64)
            pad_ids = tf.cast(tf.math.greater_equal(trim_sequence_tensor, pad), dtype=tf.int64)

        return sep_ids, tf.cast(pad_ids, dtype=tf.float32)

    
    @tf.function
    def train(input_ids, start_label, end_label, seg_ids, mask):
        
        with tf.GradientTape() as tape:
            _, _, _, output = bert(input_ids, seg_ids, mask)
            start_prediction, end_prediction = classifier(output)
            start_loss = start_loss_object(start_label, start_prediction)
            end_loss = end_loss_object(end_label, end_prediction)

            loss = (start_loss + end_loss) / 2

        variables = bert.trainable_variables + classifier.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        start_acc, _ = get_accuracy(start_label, start_prediction)
        end_acc, _ = get_accuracy(end_label, end_prediction)

        return loss, (start_acc + end_acc) / 2

    def eval():
        test_dataset = tf.data.Dataset.from_tensor_slices((test_lines, test_sep, test_mask))
        test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).repeat(1)
        predictions = {}

        for (test_step, (test_line, test_s, test_p)) in enumerate(test_dataset):
            seg_ids, pad_ids = create_masks(test_s, test_p)
        
            #5 - (a.index(3)+1)
            with tf.GradientTape() as tape:
                _, _, _, output = bert(test_line, seg_ids, pad_ids)
                start_prediction, end_prediction = classifier(output)

            start_id = tf.argmax(start_prediction, axis=-1).numpy() - test_s.numpy() -1
            end_id = tf.argmax(end_prediction, axis=-1).numpy() - test_s.numpy() -1

            test_line = test_line.numpy()
            for i, x in enumerate(start_id):
                a = list(test_line[i][x:end_id[i]])
                for k, x in enumerate(a):
                    a[k] = int(x)
                if end_id[i] > 0 and x > 0:    
                    if end_id[i] > x:
                        predictions[ids[(BATCH_SIZE*test_step)+i]] = sp.DecodeIdsWithCheck(a)
                    else:
                        predictions[ids[(BATCH_SIZE*test_step)+i]] = ''
        
        return predictions
        


    for step, data in enumerate(tqdm(dataset)):
        input_ids = data['feature0']
        start_label = data['feature1']
        end_label = data['feature2']
        seg_ids, pad_ids = create_masks(data['feature3'], data['feature4'])

        loss, train_acc = train(input_ids, start_label, end_label, seg_ids, pad_ids)

        if (step + 1) % 500 == 0:
            with writer.as_default():    
                print(f"[{version}] Training loss: {loss} | Train ACC: {train_acc}")
                tf.summary.scalar('Loss', loss, step=(step+1))
                tf.summary.scalar('Train ACC', train_acc, step=(step+1))
        if (step + 1) % 2000 == 0:
            predictions = eval()
            with open(f'./{version}/{version}-{step+1}_{lr}', 'w') as f:
                f.write(json.dumps(predictions))
            with open(f'./{version}/{version}-{step+1}_{lr}.txt', 'w') as f:
                print(predictions, file=f)

    predictions = eval()
    with open(f'./{version}/{version}_{lr}', 'w') as f:
        f.write(json.dumps(predictions))
    with open(f'./{version}/{version}_{lr}.txt', 'w') as f:
        print(predictions, file=f)


train('v1.1', 10, 48, 365, 5e-5)
train('v1.1', 10, 48, 365, 5e-4)
train('v1.1', 10, 48, 365, 1e-3)
train('v2.0', 10, 48, 814, 5e-5)
train('v2.0', 10, 48, 814, 5e-4)
train('v2.0', 10, 48, 814, 1e-3)
