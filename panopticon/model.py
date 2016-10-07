import tensorflow           as tf
import panopticon.data_sets as data_sets


def inference(inputs, is_training):
    # http://tkengo.github.io/blog/2016/03/11/understanding-convolutional-neural-networks-for-nlp/で引用されている
    # hang, Y., & Wallace, B. (2015). A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classificationを参照。

    outputs = tf.reshape(inputs, (-1, data_sets.history_size, 1, data_sets.channel_size))  # 幅を1にして、1次元データをconvolution2dできるようにします。
    outputs_list = [tf.contrib.layers.max_pool2d(tf.contrib.layers.convolution2d(outputs, 128, (kernel_size, 1), padding='VALID'), (data_sets.history_size - kernel_size + 1, 1)) for kernel_size in (5, 4, 3, 2)]
    outputs = tf.concat(1, [tf.contrib.layers.flatten(outputs) for outputs in outputs_list])  # 次元0はバッチなので、次元1でconcatします。
    outputs = tf.contrib.layers.stack(outputs, tf.contrib.layers.fully_connected, (1024, 512))
    outputs = tf.contrib.layers.dropout(outputs, is_training=is_training)

    return tf.contrib.layers.linear(outputs, 2)


def loss(logits, labels):
    result = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

    tf.scalar_summary('loss', result)

    return result;


def train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    return tf.contrib.metrics.accuracy(tf.reshape(tf.nn.top_k(logits).indices, (-1,)), labels)
