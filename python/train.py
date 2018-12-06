# -*- coding: utf-8 -*-
"""

"""

__author__ = "Gil"
__status__ = "test"
__version__ = "0.1"
__date__ = "Nov 2018"

import const
from pyfasttext import FastText
import os
import sys
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None


def predict_label(model, text):
    try:
        return str(model.predict_single(text, k=1)[0])
    except IndexError:
        return 'NEUTRAL'


def validate_model(model_file, validate_file):
    model = FastText()
    model.load_model(model_file + '.bin')

    validate = pd.read_csv(validate_file, sep='\t', names=['label', 'review'], lineterminator='\n')

    validate['predict'] = validate.apply(lambda row: predict_label(model, row['review']), axis=1)

    # all_all_count = validate.shape[0]
    good_good = validate.query(" predict == 'GOOD' and label == '__label__GOOD' ").shape[0]
    bad_bad = validate.query(" predict == 'BAD' and label == '__label__BAD' ").shape[0]
    # neutral_neutral = validate.query(" predict == 'NEUTRAL' and label == '__label__NEUTRAL' ").shape[0]

    all_good = validate.query(" label == '__label__GOOD' ").shape[0]
    all_bad = validate.query(" label == '__label__BAD' ").shape[0]
    # all_neutral = validate.query(" label == '__label__NEUTRAL' ").shape[0]

    good_all = validate.query(" predict == 'GOOD' ").shape[0]
    bad_all = validate.query(" predict == 'BAD' ").shape[0]
    # neutral_all = validate.query(" predict == 'NEUTRAL' ").shape[0]

    # micro_precision = (good_good + bad_bad + neutral_neutral)/all_all_count
    # micro_recall = micro_precision
    # micro_f1 = micro_precision
    micro_precision = good_good / good_all
    micro_recall = good_good / all_good
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    # macro_precision = ((good_good/good_all) + (bad_bad/bad_all) + (neutral_neutral/neutral_all))/3
    # macro_recall = ((good_good/all_good) + (bad_bad/all_bad) + (neutral_neutral/all_neutral))/3
    macro_precision = ((good_good / good_all) + (bad_bad / bad_all)) / 2
    macro_recall = ((good_good / all_good) + (bad_bad / all_bad)) / 2
    macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1


def train():
    train_file = const.train_processed_binary_file_name
    validate_file = const.validate_processed_binary_file_name

    current_best_score = 0
    current_best_name = ''
    lr = 0.01
    
    for epoch_i in range(1, 30):
        start_time = datetime.datetime.now().replace(microsecond=0)

        model_file_name = 'data/model_' + str(lr) + '_' + str(epoch_i)

        model = FastText()
        model.supervised(
            input=train_file,
            output=model_file_name,
            lr=lr,
            epoch=epoch_i,
            loss='softmax',
            wordNgrams=3,
            thread=12,
            ws=5,
            minn=2,
            maxn=4,
            dim=50)

        micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = validate_model(
            model_file_name, validate_file)

        end_time = datetime.datetime.now().replace(microsecond=0)

        result_log = ("epoch:" + str(epoch_i) + ': micro precision:' + str(round(micro_precision, 4)) +
                      ', micro_recall:' + str(round(micro_recall, 4)) +
                      ', micro_f1:' + str(round(micro_f1, 4)) +
                      ', macro_precision:' + str(round(macro_precision, 4)) +
                      ', macro_recall:' + str(round(macro_recall, 4)) +
                      ', macro_f1:' + str(round(macro_f1, 4)) +
                      ', lr:' + str(lr) +
                      ', duration:' + str(end_time - start_time))

        if current_best_score < micro_f1:
            current_best_score = micro_f1
            print(result_log + ' ====> Model improved!!!!')
            if current_best_name != '':
                os.remove(current_best_name)
            current_best_name = model_file_name + '.bin'

        else:
            print(result_log)
            os.remove(model_file_name + '.bin')
            os.remove(model_file_name + '.vec')

        sys.stdout.flush()


if __name__ == '__main__':

    train()
