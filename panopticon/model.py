import tensorflow as tf


def inference(inputs, is_training):
    # 1次元の畳込みは全く経験がないので、これでよいのかわかりません……。

    l1 = tf.reshape(inputs, (-1, 8, 1, 8))  # 幅を1にして、1次元データをconvolution2dできるようにします。
    
    l2_1 = tf.contrib.layers.max_pool2d(tf.contrib.layers.convolution2d(l1, 64, (4, 1), 'VALID'), (5, 2))
    l2_2 = tf.contrib.layers.max_pool2d(tf.contrib.layers.convolution2d(l1, 64, (3, 1), 'VALID'), (6, 2))
    l2_3 = tf.contrib.layers.max_pool2d(tf.contrib.layers.convolution2d(l1, 64, (2, 1), 'VALID'), (7, 2))

    outputs = tf.contrib.layers.flatten(l2_1)
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
