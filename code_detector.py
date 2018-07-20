import numpy as np
import tensorflow as tf
import helpers
import re
import os

from model import *
from sklearn.metrics import accuracy_score


all_list = list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\x0b\x0c')
unk_index = 5
batch_size = 1


def test_feed(batch_inp, batch_out):
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch_inp, max_sequence_length=1051)
    decoder_targets_, _ = helpers.batch([(sequence) + [EOS] for sequence in batch_out], max_sequence_length=1051)
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }


def index_it(content):
    content = re.sub('\d', '4', content)
    result_list = []
    for item in content:
        if item == '4':
            result_list.append(int(item))
        else:
            try:
                result_list.append(all_list.index(item) + 6)
            except ValueError:
                result_list.append(unk_index)
    return result_list


def append_to_list1(a, b):
    for item1, item2 in zip(a, b):
        if item2 == 2:
            a[item1[0]][2].append(2)
            continue
        if item2 == 3:
            a[item1[0]][2].append(3)
            continue
        a[item1[0]][2].append(0)
    return a


def append_to_list2(a, b, c):
    for item1, item2 in zip(a, b):
        if item2 == 2:
            c[item1[0]][2].append(2)
            continue
        if item2 == 3:
            c[item1[0]][2].append(3)
            continue
        c[item1[0]][2].append(0)
    return c


class Detector:
    def __init__(self):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint("./model"))  # search for checkpoint file

    def detect(self, all_data):
        main_list = []
        for index, character in enumerate(all_data):
            main_list.append((index, character, []))
        if len(main_list) <= 900:
            inp = index_it(all_data)
            fd = test_feed([inp], [inp])
            predict_ = self.sess.run(decoder_prediction, fd)
            predict = predict_.T.tolist()[0]
            main_list = append_to_list1(main_list, predict)
        else:
            main_list_len = len(main_list)
            aux_list_len_old = 0
            for i in range(0, main_list_len, 200):
                aux_list = main_list[i:(1040 + i)]
                if (len(aux_list) > 900) or ((len(aux_list) <= 900) and aux_list_len_old == 1040):
                    part_data = ''.join([x[1] for x in aux_list])
                    inp = index_it(part_data)
                    fd = test_feed([inp], [inp])
                    predict_ = self.sess.run(decoder_prediction, fd)
                    predict = predict_.T.tolist()[0]
                    main_list = append_to_list2(aux_list, predict, main_list)
                    aux_list_len_old = len(aux_list)
                else:
                    break
        text_with_mark = ''
        mark_old = 'start'
        mark = ''
        for item in main_list:
            average = sum(item[2])/len(item[2])
            if (average >= 2) and (average <= 2.05):
                mark = 'code'
            if average > 2.05:
                mark = 'text'
            if average < 2:
                mark = 'text'
            if mark_old == 'start':
                text_with_mark = '<-start_' + mark + '->\n' + item[1]
                mark_old = mark
                continue
            if mark == mark_old:
                text_with_mark += item[1]
                mark_old = mark
                continue
            if mark != mark_old:
                text_with_mark += '\n<-end_' + mark_old + '->\n'
                text_with_mark += '\n<-start_' + mark + '->\n'
                text_with_mark += item[1]
                mark_old = mark
                continue
        text_with_mark += '\n<-end_' + mark + '->\n'
        return text_with_mark

    def close_sess(self):
        self.sess.close()
