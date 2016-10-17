import os
import panopticon.model     as model
import panopticon.data_sets as data_sets
import tensorflow           as tf


def main():
    inputs = tf.placeholder(tf.float32, (None, data_sets.history_size * data_sets.channel_size))
    labels = tf.placeholder(tf.int32, (None))
    is_training = tf.placeholder(tf.bool)

    logits = model.inference(inputs, is_training)
    loss = model.loss(logits, labels)
    train = model.train(loss)
    accuracy = model.accuracy(logits, labels)

    global_step = tf.contrib.framework.get_or_create_global_step()
    inc_global_step = tf.assign(global_step, tf.add(global_step, 1))

    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()

    train_data_set, test_data_set = data_sets.load()

    if not os.path.exists('./checkpoints'):
        os.mkdir('checkpoints')

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        summary_writer = tf.train.SummaryWriter('./logs')

        checkpoint = tf.train.latest_checkpoint('./checkpoints')
        if checkpoint:
            saver.restore(session, checkpoint)

        while True:
            global_step_value = session.run(global_step)

            if global_step_value % 10 == 0:
                saver.save(session, './checkpoints/model', global_step=global_step_value)
                print('global step %5d: accuracy = %.04f.' % (global_step_value, session.run(accuracy, feed_dict={inputs: test_data_set.inputs, labels: test_data_set.labels, is_training: False})))

            inputs_value, labels_value = train_data_set.next_batch(100)
            _, summary_value = session.run((train, summary), feed_dict={inputs: inputs_value, labels: labels_value, is_training: True})

            summary_writer.add_summary(summary_value, global_step_value)
            summary_writer.flush()

            session.run(inc_global_step)


if __name__ == '__main__':
    main()
