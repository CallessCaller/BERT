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
QQP = False  # 363828 40430
COLA = False # 8550 1042
QNLI = False # 104620 5453
MNLI = False # 392575 9815
SST = False  # 67349 872
MRPC = True # 3668 408
STS = False  # 5749 1500

num_data = {
    'RTE':  2437,
    'QQP':  363828,
    'CoLA': 8550,
    'QNLI': 104620,
    'MNLI': 392575,
    'SST-2':  67349,
    'MRPC': 3668,
    'STS-B':  5749,
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

feature_description_sts = {
    'feature0': tf.io.FixedLenFeature([seq_len], tf.int64),
    'feature1': tf.io.FixedLenFeature([], tf.float32),
    'feature2': tf.io.FixedLenFeature([], tf.int64),
    'feature3': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example):
    return tf.io.parse_example(example, feature_description)

def _parse_function_sts(example):
    return tf.io.parse_example(example, feature_description_sts)

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
    BUFFER_SIZE = num_data[task] // BATCH_SIZE
    warm_up_steps = warm_up

    lr = lr

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=total_step, end_learning_rate=0.)
    lr_schedule = WarmUp(initial_learning_rate=lr, decay_schedule_fn=lr_schedule, warmup_steps=warm_up_steps)
    optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01)
    f1 = tfa.metrics.F1Score(1)
    f1_score = None

    if 'STS' in task:
        dataset = dataset.repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(_parse_function_sts)
    else:
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
        
    if 'STS' in task:
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
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
            _, _, output, _ = bert(input_ids, seg_ids, mask, False)
            prediction = classifier(output)
            loss = loss_object(label, prediction)

        variables = bert.trainable_variables + classifier.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        if task == 'CoLA':
            label = tf.one_hot(label, depth=2)
            matthew.update_state(label, prediction)
            acc = matthew.result()
        elif 'STS' in task:
            label = tf.reshape(label, [-1, 1])
            label = tf.cast(label, dtype=tf.float32)
            prediction = tf.cast(prediction, dtype=tf.float32)
            acc = tfp.stats.correlation(prediction, label)
        else:
            acc, _ = get_accuracy(label, prediction)

        return loss, acc
    
    @tf.function
    def eval(input_ids, label, seg_ids, mask):
        _, _, output, _ = bert(input_ids, seg_ids, mask, False)
        prediction = classifier(output, False)
        if task == 'CoLA':
            label = tf.one_hot(label, depth=2)
            matthew.update_state(label, prediction)
            acc = matthew.result()
        elif 'STS' in task:
            acc = prediction
        else:
            _, acc = get_accuracy(label, prediction)

            prediction = tf.cast(tf.argmax(prediction, axis=-1), dtype=tf.float32)
            prediction = tf.reshape(prediction, [-1, 1])
            #prediction = tf.cast(prediction, dtype=tf.float32)
            label = tf.reshape(label, [-1, 1])
            label = tf.cast(label, dtype=tf.float32)
            f1.update_state(label, prediction)
            f1_score = f1.result()
            return loss, (acc, f1_score)

        return loss, acc


    for step, data in enumerate(tqdm(dataset)):
        input_ids = data['feature0']
        label = data['feature1']
        seg_ids, pad_ids = create_masks(data['feature2'], data['feature3'])

        loss, train_acc = train(input_ids, label, seg_ids, pad_ids)

        if (step+1) % 100 == 0:
            f1.reset_states()
            test_dataset = tf.data.Dataset.from_tensor_slices((test_lines, test_labels, test_sep,test_mask))
            test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).repeat(1)
            if task == 'MNLI':
                test_dataset_mis = tf.data.Dataset.from_tensor_slices((test_lines_mis, test_labels_mis, test_sep_mis, test_mask_mis))
                test_dataset_mis = test_dataset_mis.batch(BATCH_SIZE, drop_remainder=False).repeat(1)
            eval_acc_total = 0
            for (test_step, (test_line, test_label, test_s, test_p)) in enumerate(test_dataset):
                seg_ids, pad_ids = create_masks(test_s, test_p)

                _, eval_acc = eval(test_line, test_label, seg_ids, pad_ids)
                if 'CoLA' not in task or 'STS' not in task:
                    eval_acc, f1_score = eval_acc
                if 'STS' in task:
                    if test_step == 0:
                        eval_acc_total = eval_acc
                    else:
                        eval_acc_total = tf.concat([eval_acc_total, eval_acc], axis=0)
                else:
                    eval_acc_total += eval_acc
            if 'MRPC' in task or 'QQP' in task:
                print(f'STEP: {step+1} | F1: {f1_score}')
                

            if task == 'CoLA':
                eval_acc_total = eval_acc
            elif 'STS' in task:
                label = tf.reshape(test_labels, [-1, 1])
                label = tf.cast(label, dtype=tf.float32)
                eval_acc_total = tf.cast(eval_acc_total, dtype=tf.float32)
                eval_acc_total = tfp.stats.correlation(eval_acc_total, label)
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
                    if 'CoLA' not in task or 'STS' not in task:
                        eval_acc, f1_score = eval_acc
                    eval_acc_total_mis += eval_acc
                eval_acc_total_mis /= len(test_labels_mis)
                if eval_acc_total_mis > best_mis:
                    best_mis = eval_acc_total_mis

            with writer.as_default():    
                #print(f"[{task}] Training loss: {loss} | Train ACC: {train_acc} | Eval ACC: {eval_acc_total}")
                if not 'STS' in task:
                    tf.summary.scalar('Eval ACC', eval_acc_total, step=(step+1))
        with writer.as_default():    
                if not 'STS' in task:
                    tf.summary.scalar('Loss', loss, step=(step+1))
                    tf.summary.scalar('Train ACC', train_acc, step=(step+1))

    eval_acc_total = 0
    for (eval_step, (test_line, test_label, test_s, test_p)) in enumerate(test_dataset):
        seg_ids, pad_ids = create_masks(test_s, test_p)

        _, eval_acc = eval(test_line, test_label, seg_ids, pad_ids)
        if 'CoLA' not in task or 'STS' not in task:
            eval_acc, f1_score = eval_acc
        if 'STS' in task:
            if eval_step == 0:
                eval_acc_total = eval_acc
            else:
                eval_acc_total = tf.concat([eval_acc_total, eval_acc], axis=0)
        else:
            eval_acc_total += eval_acc
    if task == 'CoLA':
        eval_acc_total = eval_acc
    elif 'STS' in task:
        label = tf.reshape(test_labels, [-1, 1])
        label = tf.cast(label, dtype=tf.float32)
        eval_acc_total = tf.cast(eval_acc_total, dtype=tf.float32)
        eval_acc_total = tfp.stats.correlation(eval_acc_total, label)
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
            if 'CoLA' not in task or 'STS' not in task:
                eval_acc, f1_score = eval_acc
            eval_acc_total_mis += eval_acc
        eval_acc_total_mis /= len(test_labels_mis)
        if eval_acc_total_mis > best_mis:
            best_mis = eval_acc_total_mis

   

    with writer.as_default():    
        #print(f"[{task}] Training loss: {loss} | Train ACC: {train_acc} | Eval ACC: {eval_acc_total}")
        if not 'STS' in task:
            tf.summary.scalar('Loss', loss, step=(step+1))
            tf.summary.scalar('Train ACC', train_acc, step=(step+1))
            tf.summary.scalar('Eval ACC', eval_acc_total, step=(step+1))

    if task=='MNLI':
        print(f'matched: {best} | mismathced: {best_mis}')
        return best, best_mis
    print(f'best: {best}')
    if 'MRPC' in task or 'QQP' in task:
        return best, f1_score
    return best

RTE_best = 0
MRPC_best = 0
MRPC_best_f1 = 0
COLA_best = 0
MNLI_best = 0
MNLI_best_mis = 0
QNLI_best = 0
QQP_best = 0
QQP_best_f1 = 0
SST_best = 0

if RTE:
    best1 = tuning('RTE', 2, 32, 4, 200, 3e-4)
    best2 = tuning('RTE', 2, 32, 4, 200, 1e-4)
    best3 = tuning('RTE', 2, 32, 4, 200, 3e-5)
    best4 = tuning('RTE', 2, 32, 4, 200, 5e-5)

    RTE_best = max([best1, best2, best3, best4])
if MRPC:
    best1, f1_score1 = tuning('MRPC', 2, 16, 10, 200, 5e-4)
    print(f1_score1)

    best3, f1_score3 = tuning('MRPC', 2, 16, 10, 200, 1e-5)
    print(f1_score3)

    best4, f1_score4 = tuning('MRPC', 2, 16, 10, 200, 1e-3)
    print(f1_score4)

    MRPC_best = max([best1, best3, best4])
    #MRPC_best_f1 = max([f1_score3, f1_score4])

    print(MRPC_best_f1)

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

if STS:
    best1 = tuning('STS-B', 1, 64, 20, 80, 1e-3)
    best2 = tuning('STS-B', 1, 64, 20, 100, 1e-3)
    best3 = tuning('STS-B', 1, 32, 20, 214, 1e-3)
    best4 = tuning('STS-B', 1, 16, 10, 214, 1e-3)


    QQP_best = max([best1, best2, best3, best4])

if MNLI:
    best1, best_mis1 = tuning('MNLI', 3, 64, 4, 500, 8e-4)
    best2, best_mis2 = tuning('MNLI', 3, 64, 4, 1500, 8e-4)
    best3, best_mis3 = tuning('MNLI', 3, 64, 4, 1000, 8e-4)
    best4, best_mis4 = tuning('MNLI', 3, 64, 4, 1000, 1e-3)
    best5, best_mis5 = tuning('MNLI', 3, 64, 4, 500, 1e-3)
    best6, best_mis6 = tuning('MNLI', 3, 64, 4, 1500, 1e-3)
    best7, best_mis7 = tuning('MNLI', 3, 128, 4, 1000, 1e-3)
    best8, best_mis8 = tuning('MNLI', 3, 128, 4, 1000, 8e-4)
    best9, best_mis9 = tuning('MNLI', 3, 256, 4, 1000, 8e-4)
    best0, best_mis0 = tuning('MNLI', 3, 256, 4, 1000, 1e-3)

    MNLI_best = max([best1, best2, best3, best4, best5, best6, best7, best8, best9, best0])
    MNLI_best_mis = max([best_mis1, best_mis2, best_mis3, best_mis4, best_mis5, best_mis6, best_mis7, best_mis8, best_mis9, best_mis0])

if QNLI:
    best1 = tuning('QNLI', 2, 32, 4, 1986, 5e-4)
    best2 = tuning('QNLI', 2, 32, 4, 1986, 3e-4)

    QNLI_best = max([best1, best2])

if SST:
    best1 = tuning('SST-2', 2, 16, 4, 1256, 3e-4)
    best2 = tuning('SST-2', 2, 16, 4, 1256, 1e-4)
    best3 = tuning('SST-2', 2, 16, 4, 1256, 3e-5)
    best4 = tuning('SST-2', 2, 16, 4, 1256, 5e-5)

    SST_best = max([best1, best2, best3, best4])

if QQP:
    best1, f1_score1 = tuning('QQP', 2, 128, 4, 1000, 5e-4)
    print(f1_score1)
    best2, f1_score2 = tuning('QQP', 2, 128, 4, 1000, 3e-4)
    print(f1_score2)
    best2, f1_score3 = tuning('QQP', 2, 256, 4, 1000, 5e-4)
    print(f1_score3)
    best2, f1_score4 = tuning('QQP', 2, 64, 4, 1000, 5e-4)
    print(f1_score4)

    QQP_best = max([best1, best2])
    #QQP_best_f1 = max([f1_score1, f1_score2])
    print(QQP_best_f1)

print(f'CoLA: {COLA_best} | RTE: {RTE_best} | MRPC (F1/ACC): {MRPC_best_f1} / {MRPC_best} | MNLI: m {MNLI_best} mm {MNLI_best_mis} | QNLI: {QNLI_best} | QQP (F1/ACC): {QQP_best_f1} / {QQP_best} | SST: {SST_best} |')