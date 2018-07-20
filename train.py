import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

tf.reset_default_graph()

from model import *

f1_ = open('loss_test_track.json', 'w')
f2_ = open('loss_train_track.json', 'w')
batch_size = 100
max_batches = 11801
batches_for_test = 100
batches_for_print = 10
epoch_num = 3
loss_train_track = []
loss_test_track = []


def batch_gen():
    result1 = []
    result2 = []
    file_ = codecs.open('train.json', encoding='utf-8', mode='r')
    for line_ in file_:
        l1 = json.loads(line_)[0]
        l2 = json.loads(line_)[1]
        result1.append(l1)
        result2.append(l2)
        if len(result1) == batch_size:
            yield result1, result2
            result1 = []
            result2 = []


def next_feed(batches):
    batch = next(batches)
    batch_inp = batch[0]
    batch_out = batch[1]
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch_inp, max_sequence_length=1051)
    decoder_targets_, _ = helpers.batch([(sequence) + [EOS] for sequence in batch_out], max_sequence_length=1051)
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }


def test_feed():
    result1 = []
    result2 = []
    file_ = codecs.open('test.json', encoding='utf-8', mode='r')
    for line_ in file_:
        l1 = json.loads(line_)[0]
        l2 = json.loads(line_)[1]
        result1.append(l1)
        result2.append(l2)
        if len(result1) == batch_size:
            encoder_inputs_, encoder_input_lengths_ = helpers.batch(result1, max_sequence_length=1051)
            decoder_targets_, _ = helpers.batch([(sequence) + [EOS] for sequence in result2], max_sequence_length=1051)

            result1 = []
            result2 = []

            yield {
                encoder_inputs: encoder_inputs_,
                encoder_inputs_length: encoder_input_lengths_,
                decoder_targets: decoder_targets_,
            }


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        print('--------------------------------------------------------------\n')
        batches = batch_gen()
        for batch in range(max_batches):
            fd = next_feed(batches)
            _, l = sess.run([train_op, loss], fd)
            l = np.float32(l).item()
            loss_train_track.append(l)

            if batch == 0 or batch % batches_for_print == 0:
                print("epoch number ", epoch + 1)
                print('batch {}'.format(batch))
                print(' loss on train data: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, targ, pred) in enumerate(zip(fd[encoder_inputs].T, fd[decoder_targets].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    target    > {}'.format(targ))
                    print('    predicted > {}'.format(pred))
                    print('    accuracy > {}'.format(accuracy_score(targ, pred)))
                    if i >= 4:
                        break

            if batch == 0 or batch % batches_for_test == 0:
                batches_test = test_feed()
                test_batch = next(batches_test)
                l_ = sess.run(loss, test_batch)
                l_ = np.float32(l_).item()
                loss_test_track.append(l_)
                print()
                print('loss on test data: ', l_)
                print()
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in path: %s" % save_path)

if loss_test_track:
    json.dump(loss_test_track, f1_)

if loss_train_track:
    json.dump(loss_train_track, f2_)
