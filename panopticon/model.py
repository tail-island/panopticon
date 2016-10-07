import tensorflow as tf


def inference(inputs, is_training):
    # 1次元の畳込みは全く経験がないので、これでよいのかわかりません……。

    outputs = tf.reshape(inputs, (-1, 8, 1, 8))  # 幅を1にして、1次元データをconvolution2dできるようにします。
    outputs = tf.contrib.layers.convolution2d(outputs,  64, (4, 1))
    outputs = tf.contrib.layers.convolution2d(outputs, 128, (4, 1))
    outputs = tf.contrib.layers.max_pool2d(outputs, (2, 1))

    outputs = tf.contrib.layers.flatten(outputs)
    outputs = tf.contrib.layers.fully_connected(outputs, 1024)
    outputs = tf.contrib.layers.fully_connected(outputs,  512)

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
