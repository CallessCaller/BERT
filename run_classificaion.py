from bert import PretrainerBERT
from bert_classifier import ClassifierBERT
import datetime
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
import numpy as np
import pickle

from optimization import WarmUp, AdamWeightDecay
from tqdm import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_probability as tfp

RTE = False  # 2437 270
QQP = True  # 363828 40430
COLA = False # 8550 1042
QNLI = True # 104620 5453
MNLI = True # 392575 9815
SST = True  # 67349 872
MRPC = False # 3668 408
STS = True  # 5749 1500

num_data = {
    'RTE':  2437,
    'QQP':  363828,
    'CoLA': 8550,
    'QNLI': 104620,
    'MNLI': 392575,
    'SST-B':  67349,
    'MRPC': 3668,
    'STS-2':  5749,
}

path = '../data/GLUE'
hidden_size = 128
dropout_rate = 0.1
num_heads = 2
num_layers = 2
dff = 512
vocab_size = 30000
seq_len = 256

feature_description = {
    'feature0': tf.io.FixedLenFeature([seq_len], tf.int64),
    'feature1': tf.io.FixedLenFeature([], tf.int64),
    'feature2': tf.io.FixedLenFeature([], tf.int64),
    'feature3': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example):
    return tf.io.parse_example(example, feature_description)

@tf.function
def get_accuracy(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=-1))
    accuracies = tf.cast(accuracies, tf.float32)
    
    return tf.reduce_mean(accuracies), tf.reduce_sum(accuracies)

matthew = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)

def tuning(task, num_class, batch_size, epochs, warm_up, lr):
    print(f'Doing {task}...')
    best = 0
    bert = PretrainerBERT(num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)
    optimizer = tfa.optimizers.LAMB(learning_rate=0.00176, weight_decay_rate=0.01)
    checkpoint_path = "./checkpoints/train/ckpt-3"
    ckpt = tf.train.Checkpoint(model=bert, optimizer=optimizer)
    ckpt.read(checkpoint_path).expect_partial()

    print('Latest checkpoint restored!!')

    #model = ClassifierBERT(bert, 2, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)
    classifier = ClassifierBERT(num_class, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"./{task}/logs/" + current_time + '/train')

    dataset = tf.data.TFRecordDataset(f'{path}/{task}/train.tfrecord')
    
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    total_step = (num_data[task]*EPOCHS)//BATCH_SIZE
    BUFFER_SIZE = 50
    warm_up_steps = warm_up

    lr = lr

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=total_step, end_learning_rate=0.)
    lr_schedule = WarmUp(initial_learning_rate=lr, decay_schedule_fn=lr_schedule, warmup_steps=warm_up_steps)
    optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01)

    dataset = dataset.repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(_parse_function)

    if task=='MNLI':
        with open(f'{path}/{task}/dev_matched.pickle', 'rb') as f:
            test_lines = pickle.load(f)
            test_labels = pickle.load(f)
            test_sep = pickle.load(f)
            test_mask = pickle.load(f)

        with open(f'{path}/{task}/dev_mismatched.pickle', 'rb') as f:
            test_lines_mis = pickle.load(f)
            test_labels_mis = pickle.load(f)
            test_sep_mis = pickle.load(f)
            test_mask_mis = pickle.load(f)
    else:
        with open(f'{path}/{task}/dev.pickle', 'rb') as f:
            test_lines = pickle.load(f)
            test_labels = pickle.load(f)
            test_sep = pickle.load(f)
            test_mask = pickle.load(f) 
        
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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
    def train(input_ids, label, seg_ids, mask):
        
        with tf.GradientTape() as tape:
            _, _, output = bert(input_ids, seg_ids, mask)
            prediction = classifier(output)
            loss = loss_object(label, prediction)

        variables = bert.trainable_variables + classifier.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        if task == 'CoLA':
            label = tf.one_hot(label, depth=2)
            matthew.update_state(label, prediction)
            acc = matthew.result()
        else:
            acc, _ = get_accuracy(label, prediction)

        return loss, acc
    
    @tf.function
    def eval(input_ids, label, seg_ids, mask):
        _, _, output = bert(input_ids, seg_ids, mask, False)
        prediction = classifier(output)
        if task == 'CoLA':
            label = tf.one_hot(label, depth=2)
            matthew.update_state(label, prediction)
            acc = matthew.result()
        else:
            _, acc = get_accuracy(label, prediction)

        return loss, acc


    for step, data in enumerate(dataset):
        input_ids = data['feature0']
        label = data['feature1']
        seg_ids, pad_ids = create_masks(data['feature2'], data['feature3'])

        loss, train_acc = train(input_ids, label, seg_ids, pad_ids)

        if (step+1) % 100 == 0:
            test_dataset = tf.data.Dataset.from_tensor_slices((test_lines, test_labels, test_sep,test_mask))
            test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).repeat(1)
            if task == 'MNLI':
                test_dataset_mis = tf.data.Dataset.from_tensor_slices((test_lines_mis, test_labels_mis, test_sep_mis, test_mask_mis))
                test_dataset_mis = test_dataset_mis.batch(BATCH_SIZE, drop_remainder=False).repeat(1)
            eval_acc_total = 0
            for (test_step, (test_line, test_label, test_s, test_p)) in enumerate(test_dataset):
                seg_ids, pad_ids = create_masks(test_s, test_p)

                _, eval_acc = eval(test_line, test_label, seg_ids, pad_ids)
                eval_acc_total += eval_acc

            if task == 'CoLA':
                eval_acc_total = eval_acc
            else:
                eval_acc_total /= len(test_labels)

            if eval_acc_total > best:
                best = eval_acc_total

            if task == 'MNLI':
                best_mis = 0
                eval_acc_total_mis = 0
                for (test_step, (test_line, test_label, test_s, test_p)) in enumerate(test_dataset_mis):
                    seg_ids, pad_ids = create_masks(test_s, test_p)

                    _, eval_acc = eval(test_line, test_label, seg_ids, pad_ids)
                    eval_acc_total_mis += eval_acc
                eval_acc_total_mis /= len(test_labels_mis)
                if eval_acc_total_mis > best_mis:
                    best_mis = eval_acc_total_mis

            with writer.as_default():    
                print(f"{task} Training loss: {loss} | Train ACC: {train_acc} | Eval ACC: {eval_acc_total}")
                tf.summary.scalar('Eval ACC', eval_acc_total, step=(step+1))
        with writer.as_default():    
                tf.summary.scalar('Loss', loss, step=(step+1))
                tf.summary.scalar('Train ACC', train_acc, step=(step+1))

    eval_acc_total = 0
    for (eval_step, (test_line, test_label, test_s, test_p)) in enumerate(test_dataset):
        seg_ids, pad_ids = create_masks(test_s, test_p)

        _, eval_acc = eval(test_line, test_label, seg_ids, pad_ids)
        eval_acc_total += eval_acc
    if task == 'CoLA':
        eval_acc_total = eval_acc
    else:
        eval_acc_total /= len(test_labels)
    if eval_acc_total > best:
        best = eval_acc_total

    if task == 'MNLI':
        best_mis = 0
        eval_acc_total_mis = 0
        for (test_step, (test_line, test_label, test_s, test_p)) in enumerate(test_dataset_mis):
            seg_ids, pad_ids = create_masks(test_s, test_p)

            _, eval_acc = eval(test_line, test_label, seg_ids, pad_ids)
            eval_acc_total_mis += eval_acc
        eval_acc_total_mis /= len(test_labels_mis)
        if eval_acc_total_mis > best_mis:
            best_mis = eval_acc_total_mis

   

    with writer.as_default():    
        print(f"{task} Training loss: {loss} | Train ACC: {train_acc} | Eval ACC: {eval_acc_total}")
        tf.summary.scalar('Loss', loss, step=(step+1))
        tf.summary.scalar('Train ACC', train_acc, step=(step+1))
        tf.summary.scalar('Eval ACC', eval_acc_total, step=(step+1))

    if task=='MNLI':
        print(f'matched: {best} | mismathced: {best_mis}')
        return best, best_mis
    print(f'best: {best}')
    return best

RTE_best = 0
MRPC_best = 0
COLA_best = 0
MNLI_best = 0
MNLI_best_mis = 0
QNLI_best = 0
QQP_best = 0
SST_best = 0

if RTE:
    best1 = tuning('RTE', 2, 32, 4, 200, 3e-4)
    best2 = tuning('RTE', 2, 32, 4, 200, 1e-4)
    best3 = tuning('RTE', 2, 32, 4, 200, 3e-5)
    best4 = tuning('RTE', 2, 32, 4, 200, 5e-5)

    RTE_best = max([best1, best2, best3, best4])
if MRPC:
    best1 = tuning('MRPC', 2, 32, 4, 200, 3e-4)
    best2 = tuning('MRPC', 2, 32, 4, 200, 1e-4)
    best3 = tuning('MRPC', 2, 32, 4, 200, 3e-5)
    best4 = tuning('MRPC', 2, 32, 4, 200, 5e-5)

    MRPC_best = max([best1, best2, best3, best4])
if COLA:
    matthew.reset_states()
    best1 = tuning('CoLA', 2, 16, 4, 320, 3e-4)
    matthew.reset_states()
    best2 = tuning('CoLA', 2, 16, 4, 320, 1e-4)
    matthew.reset_states()
    best3 = tuning('CoLA', 2, 16, 4, 320, 3e-5)
    matthew.reset_states()
    best4 = tuning('CoLA', 2, 16, 4, 320, 5e-5)

    COLA_best = max([best1, best2, best3, best4])

if MNLI:
    best1, best_mis1 = tuning('MNLI', 3, 128, 4, 1000, 3e-4)
    best2, best_mis2 = tuning('MNLI', 3, 128, 4, 1000, 1e-4)
    best3, best_mis3 = tuning('MNLI', 3, 128, 4, 1000, 3e-5)
    best4, best_mis4 = tuning('MNLI', 3, 128, 4, 1000, 5e-5)

    MNLI_best = max([best1, best2, best3, best4])
    MNLI_best_mis = max([best_mis1, best_mis2, best_mis3, best_mis4])

if QNLI:
    best1 = tuning('QNLI', 2, 32, 4, 1986, 3e-4)
    best2 = tuning('QNLI', 2, 32, 4, 1986, 1e-4)
    best3 = tuning('QNLI', 2, 32, 4, 1986, 3e-5)
    best4 = tuning('QNLI', 2, 32, 4, 1986, 5e-5)

    QNLI_best = max([best1, best2, best3, best4])

if SST:
    best1 = tuning('SST-2', 2, 16, 4, 1256, 3e-4)
    best2 = tuning('SST-2', 2, 16, 4, 1256, 1e-4)
    best3 = tuning('SST-2', 2, 16, 4, 1256, 3e-5)
    best4 = tuning('SST-2', 2, 16, 4, 1256, 5e-5)

    SST_best = max([best1, best2, best3, best4])

if QQP:
    best1 = tuning('QQP', 2, 128, 4, 1000, 3e-4)
    best2 = tuning('QQP', 2, 128, 4, 1000, 1e-4)
    best3 = tuning('QQP', 2, 128, 4, 1000, 3e-5)
    best4 = tuning('QQP', 2, 128, 4, 1000, 5e-5)

    QQP_best = max([best1, best2, best3, best4])

print(f'CoLA: {COLA_best} | RTE: {RTE_best} | MRPC: {MRPC_best} | MNLI: m {MNLI_best} mm {MNLI_best_mis} | QNLI: {QNLI_best} | QQP: {QQP_best} | SST: {SST_best} |')