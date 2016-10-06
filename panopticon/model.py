import tensorflow as tf


def inference(inputs, is_training):
    outputs = tf.reshape(inputs, (-1, 8, 8))

    # 1次元の畳込みは、contrib.layersにありませんでした……。辛い。
    # 1次元の畳込みは全く経験がないので、これでよいのかもわかりません……。

    w1 = tf.Variable(tf.truncated_normal((4, 8, 64)))
    b1 = tf.Variable(tf.constant(0.1, shape=(64,)))
    outputs = tf.nn.relu(tf.add(tf.nn.conv1d(outputs, w1, 1, 'SAME'), b1))

    w2 = tf.Variable(tf.truncated_normal((4, 64, 128)))
    b2 = tf.Variable(tf.constant(0.1, shape=(128,)))
    outputs = tf.nn.relu(tf.add(tf.nn.conv1d(outputs, w2, 1, 'SAME'), b2))

    outputs = tf.reshape(outputs, (-1, 8 * 128))

    outputs = tf.contrib.layers.fully_connected(outputs, 1024)
    outputs = tf.contrib.layers.fully_connected(outputs, 512)
    outputs = tf.contrib.layers.dropout(outputs, is_training=is_training)
    outputs = tf.contrib.layers.linear(outputs, 3)

    return outputs


def loss(logits, labels):
    result = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

    tf.scalar_summary('loss', result)

    return result;


def train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    return tf.contrib.metrics.accuracy(tf.reshape(tf.nn.top_k(logits).indices, (-1,)), labels)
