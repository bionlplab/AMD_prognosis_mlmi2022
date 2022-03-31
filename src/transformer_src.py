from re import X
import tensorflow as tf
import numpy as np

class TransformerEncoderToken(tf.keras.Model):
    def __init__(self, config):
        super(TransformerEncoderToken, self).__init__()

        self.model_dim = config["model_dim"]
        self.num_layers = config["num_layers"]
        self.pos_encoding = positional_encoding((config["maximum_seq_length"]*2)+1, self.model_dim)
        self.dense = tf.keras.layers.Dense(units=self.model_dim, activation="relu")
        self.enc_layers = [EncoderLayer(config["model_dim"], config["num_heads"], 
            config["intermediate_dim"], config["dropout_rate"]) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(config["dropout_rate"])
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        if config["use_sep_token"]:
            self.use_sep_token = True
            self.sep_token = tf.keras.layers.Embedding(1, self.model_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(0., 1.))
        else:
            self.use_sep_token = False

        if config["use_pred_token"]:
            self.use_pred_token = True
            self.pred_token = tf.keras.layers.Embedding(1, self.model_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(0., 1.))
        else:
            self.use_pred_token = False

    def call(self, x, training, mask):
        """
        x: (batch_size, max_seq_len, feature_dim)
        """
        # image feature dim to model dim
        x = self.dense(x) # (batch_size, max_seq_len, model_dim)

        if self.use_sep_token:
            sep_tensor = self.sep_token(0)
            sep_tensor = tf.reshape(sep_tensor, (1,1,-1))
            sep_tensor = tf.tile(sep_tensor, (x.shape[0],x.shape[1],1))
            x = tf.reshape(tf.concat([x, sep_tensor], axis=-1), shape=(x.shape[0], x.shape[1]*2, x.shape[2]))
            mask = tf.tile(tf.reshape(mask, shape=(mask.shape[0], mask.shape[-1], 1)), (1,1,2))
            mask = tf.reshape(mask, shape=(mask.shape[0], 1, 1, mask.shape[1]*2))

        if self.use_pred_token:
            pred_tensor = self.pred_token(0)
            pred_tensor = tf.reshape(pred_tensor, (1,1,-1))
            pred_tensor = tf.tile(pred_tensor, (x.shape[0],1,1))
            x = tf.concat([pred_tensor, x], axis=1)
            mask = tf.concat([tf.zeros(shape=(mask.shape[0],1,1,1)), mask], axis=-1)

        # add positional encoding

        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32)) # scaling embedding
        x += self.pos_encoding[:, :x.shape[1], :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask) # (batch_size, input_seq_len, d_model)

        return self.prediction(x)  # (batch_size, input_seq_len, 1)

class TransformerEncoderSegment(tf.keras.Model):
    def __init__(self, config):
        super(TransformerEncoderSegment, self).__init__()

        self.model_dim = config["model_dim"]
        self.num_layers = config["num_layers"]
        self.pos_encoding = positional_encoding(config["maximum_seq_length"], self.model_dim)
        self.dense = tf.keras.layers.Dense(units=self.model_dim, activation="relu")
        self.enc_layers = [EncoderLayer(config["model_dim"], config["num_heads"], 
            config["intermediate_dim"], config["dropout_rate"]) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(config["dropout_rate"])
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        if config["use_segment_embedding"] == "bisegment":
            self.use_bisegment_embedding = True
            self.segment_embedding = tf.keras.layers.Embedding(2, self.model_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(0., 1.))
        else:
            self.use_bisegment_embedding = False
        if config["use_segment_embedding"] == "separate":
            self.use_separate_segment_embedding = True
            self.segment_embedding = tf.keras.layers.Embedding(config["maximum_seq_length"], self.model_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(0., 1.))
        else:
            self.use_separate_segment_embedding = False

    def call(self, x, segment_x, training, mask):
        """
        x: (batch_size, max_seq_len, feature_dim)
        """
        # image feature dim to model dim
        x = self.dense(x) # (batch_size, max_seq_len, model_dim)
        segment_tensor = self.segment_embedding(segment_x)

        # add positional encoding and segment embedding
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32)) # scaling embedding
        x += self.pos_encoding[:, :x.shape[1], :]
        x += segment_tensor
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask) # (batch_size, input_seq_len, d_model)

        return self.prediction(x)  # (batch_size, input_seq_len, 1)

class TransformerEncoder(tf.keras.Model):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.model_dim = config["model_dim"]
        self.num_layers = config["num_layers"]
        self.pos_encoding = positional_encoding(config["maximum_seq_length"], self.model_dim)
        self.dense = tf.keras.layers.Dense(units=self.model_dim, activation="relu")
        self.enc_layers = [EncoderLayer(config["model_dim"], config["num_heads"], 
            config["intermediate_dim"], config["dropout_rate"]) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(config["dropout_rate"])
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def call(self, x, training, mask):
        """
        x: (batch_size, seq_len, feature_dim)
        """
        this_seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.dense(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32)) # scaling embedding
        x += self.pos_encoding[:, :this_seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask) # (batch_size, input_seq_len, d_model)

        return self.prediction(x)  # (batch_size, input_seq_len, 1)

class TransformerEncoderExperiment(tf.keras.Model):
    def __init__(self, config):
        super(TransformerEncoderExperiment, self).__init__()

        self.model_dim = config["model_dim"]
        self.num_layers = config["num_layers"]
        self.pos_encoding = positional_encoding(config["maximum_seq_length"], self.model_dim)
        self.dense = tf.keras.layers.Dense(units=self.model_dim, activation="relu")
        self.enc_layers = [EncoderLayer(config["model_dim"], config["num_heads"], 
            config["intermediate_dim"], config["dropout_rate"]) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(config["dropout_rate"])
        self.prediction = tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def call(self, x, training, mask):
        """
        x: (batch_size, seq_len, feature_dim)
        """
        this_seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        pos_x = self.pos_encoding[:, :this_seq_len, :]
        pos_x = tf.tile(pos_x, (x.shape[0], 1, 1))
        x = tf.concat([x, pos_x], axis=-1)
        x = self.dense(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32)) # scaling embedding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask) # (batch_size, input_seq_len, d_model)

        return self.prediction(x)  # (batch_size, input_seq_len, 1)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, intermediate_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.pw_ffn = PositionWiseFeedForwardNet(model_dim, intermediate_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        mha_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        mha_output = self.dropout1(mha_output, training=training)
        out1 = self.layernorm1(x + mha_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.pw_ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "dimension of the model must be multiplicative of number of heads"
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.per_head_dim = model_dim / num_heads

        self.W_q = tf.keras.layers.Dense(model_dim, use_bias=False)
        self.W_k = tf.keras.layers.Dense(model_dim, use_bias=False)
        self.W_v = tf.keras.layers.Dense(model_dim, use_bias=False)
        self.W_O = tf.keras.layers.Dense(model_dim, use_bias=False)

    def split_heads(self, x, batch_size):
        """Split the total dimension of the tensor into number of heads x each dimension"""
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads), int(self.per_head_dim)))
        
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.W_q(q)  # (batch_size, seq_len, d_model)
        k = self.W_k(k)  # (batch_size, seq_len, d_model)
        v = self.W_v(v)  # (batch_size, seq_len, d_model)    

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # shape of attention = (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, d_model)
        output = self.W_O(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights

class PositionWiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, model_dim, intermediate_dim):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.ffn1 = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(model_dim)
    
    def call(self, x):
        x = self.ffn1(x)
        return self.ffn2(x)

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: (..., seq_len_q, depth)
        k: (..., seq_len_k, depth)
        v: (..., seq_len_v, depth)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = tf.divide(matmul_qk, tf.math.sqrt(dk))

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  ### why multiply 1e9?

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v) 

    return output, attention_weights

def positional_encoding(position, model_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(model_dim)[np.newaxis, :], model_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(position, i, model_dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(model_dim))
    return position * angle_rates

