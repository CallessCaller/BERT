import sentencepiece
import tensorflow as tf
from tensorflow.keras import layers, models


class PretrainerBERT(models.Model):
    def __init__(self, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dff = dff
        self.num_heads = num_heads

        self.bert = BERT(self.num_layers, self.vocab_size, self.seq_len, self.hidden_size, self.dff, self.num_heads, dropout_rate)
        self.dense_for_nsp = layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.dense_for_mlm = layers.Dense(self.vocab_size, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, input_ids, seg_ids, mask, training=True):
        x = self.bert(input_ids, seg_ids, mask, training)

        nsp_prediction = self.dense_for_nsp(x[:,0])
        mlm_prediction = self.dense_for_mlm(x)

        return mlm_prediction, nsp_prediction


class BERT(layers.Layer):
    def __init__(self, num_layers, vocab_size, seq_len, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dff = dff
        self.num_heads = num_heads

        self.embedding_layer = EmbeddingProcessor(self.vocab_size, self.seq_len, self.hidden_size, dropout_rate)
        self.transformers = Transformer(self.num_layers, self.hidden_size, self.dff, self.num_heads, dropout_rate)

    def call(self, input_ids, seg_ids, attn_mask, training=True):
        x = self.embedding_layer(input_ids, seg_ids, training)
        x = self.transformers(x, attn_mask, training)
        
        return x


class EmbeddingProcessor(layers.Layer):
    def __init__(self, vocab_size, seq_len, hidden_size, dropout_rate=0.1, initialize_range=0.02):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.seq_list = tf.reshape(tf.range(self.seq_len), [1, -1])

        self.position_embedding = layers.Embedding(input_dim=self.seq_len, 
                                                   output_dim=self.hidden_size, 
                                                   embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initialize_range))
        self.segment_embedding  = layers.Embedding(input_dim=2, 
                                                   output_dim=self.hidden_size, 
                                                   embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initialize_range))
        self.embedding          = layers.Embedding(input_dim=self.vocab_size, 
                                                   output_dim=self.hidden_size, 
                                                   embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initialize_range))
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, input_ids, seg_ids, training=True):
        embedding = self.embedding(input_ids)

        embedding += self.position_embedding(self.seq_len)
        embedding += self.segment_embedding(seg_ids)

        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding, training=training)

        return embedding


class Transformer(layers.Layer):
    def __init__(self, num_layers, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.encoders = [TransformerBlock(self.hidden_size, dff, self.num_heads, dropout_rate) for _ in range(num_layers)]

    def call(self, x, att_mask, training=True):
        for i in range(self.num_layers):
            x = self.encoders[i](x, att_mask, training)

        return x


class TransformerBlock(layers.Layer):
    def __init__(self, hidden_size, dff, num_heads, dropout_rate=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dff = dff

        self.attention_layer = AttentionLayer(self.hidden_size, self.num_heads, dropout_rate)
        self.projection_layer_1 = ProjectionLayer(self.hidden_size, dropout_rate)
        self.projection_layer_2 = ProjectionLayer(self.hidden_size, dropout_rate)
        self.point_wise_feed_forward = layers.Dense(self.dff, activation='gelu', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    
    def call(self, x, att_mask, training=True):
        attention_output = self.attention_layer(x, att_mask, training)
        x = self.projection_layer_1(attention_output, x, training)

        intermidiate = self.point_wise_feed_forward(x)
        x = self.projection_layer_2(intermidiate, x, training)

        return x


class AttentionLayer(layers.Layer):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = self.hidden_size // self.num_heads

        self.wq = layers.Dense(hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.wk = layers.Dense(hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.wv = layers.Dense(hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, att_mask, training=True):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = tf.transpose(tf.reshape(q, [batch_size, -1, self.num_heads, self.depth]), perm=[0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [batch_size, -1, self.num_heads, self.depth]), perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [batch_size, -1, self.num_heads, self.depth]), perm=[0, 2, 1, 3])

        attention_scores = tf.einsum('bnqd,bnkd->bnqk')
        attention_scores = attention_scores / tf.sqrt(float(self.depth))
        
        # {1, 0} -> {0.0, -inf}
        att_mask = (1.0 - tf.expand_dims(att_mask, 1)) * -10000.
        attention_scores = tf.add(attention_scores, att_mask)

        # [b, n, q, k]
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, shape=[batch_size, -1, self.hidden_size])

        return output


class ProjectionLayer(layers.Layer):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.dense = layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()


    def call(self, output, residual, training=True):
        output = self.dense(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + residual)

        return output