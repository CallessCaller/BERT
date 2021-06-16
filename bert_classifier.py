import tensorflow as tf
from tensorflow.python.ops.array_ops import unstack
import tensorflow_addons as tfa
from bert import PretrainerBERT

class ClassifierBERTv2(tf.keras.models.Model):
    def __init__(self, num_class, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc = tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, prediction, training=True):
        prediction = self.dropout(prediction, training=training)
        prediction = self.fc(prediction)

        return prediction

class ClassifierBERT(tf.keras.models.Model):
    def __init__(self, num_class, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()

        self.dense = tf.keras.layers.Dense(hidden_size, activation='tanh', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc = tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, prediction, training=True):
        prediction = self.dense(prediction)
        #prediction = tf.nn.tanh(prediction)
        prediction = self.dropout(prediction, training=training)
        prediction = self.fc(prediction)

        return prediction

class SquadBERT(tf.keras.models.Model):
    def __init__(self, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()

        self.dense = tf.keras.layers.Dense(hidden_size, activation='gelu', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc = tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, prediction, training=True):
        prediction = self.dense(prediction)
        prediction = self.dropout(prediction, training=training)
        prediction = self.fc(prediction)

        prediction = tf.transpose(prediction, perm=[2, 0, 1])

        return prediction[0], prediction[1]