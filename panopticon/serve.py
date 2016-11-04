import RPi.GPIO as gpio
import numpy
import panopticon.data_sets as data_sets
import panopticon.data_source as data_source
import panopticon.model as model
import tensorflow as tf

from collections import deque


inputs = tf.placeholder(tf.float32, (None, data_sets.history_size * data_sets.channel_size))
is_training = tf.placeholder_with_default(False, ())

top_k = tf.nn.top_k(model.inference(inputs, is_training))

session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver.restore(session, tf.train.latest_checkpoint('./checkpoints'))


def classify(action):
    return session.run(top_k, feed_dict={inputs: numpy.asarray((action,))}).indices[0][0]


if __name__ == '__main__':
    try:
        gpio.setmode(gpio.BCM)
        gpio.setup(18, gpio.OUT)
        gpio.output(18, gpio.HIGH)

        pwm = gpio.PWM(18, 100)
        pwm.start(0)

        action_classes = deque((), 10)
        
        for action_class in map(classify, data_sets.actions(data_source.read_poses())):
            action_classes.append(action_class)
            pwm.ChangeDutyCycle(action_classes.count(1) * 10)

    finally:
        pwm.stop()
        gpio.cleanup()
