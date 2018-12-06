# -*- coding: utf-8 -*-
"""

"""

__author__ = "Gil"
__status__ = "test"
__version__ = "0.1"
__date__ = "Nov 2018"

import const
import numpy as np
import re
import logging
import sys
import pandas as pd
pd.options.mode.chained_assignment = None

FORMAT = const.LOG_MSG_FORMAT
logging.basicConfig(format=FORMAT, datefmt=const.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def business_json_to_rows():
    logger.info('start process:' + sys._getframe().f_code.co_name)

    business_raw = pd.read_json(const.business_json_file_name, lines=True)
    business_raw = business_raw[
        ['business_id', 'name', 'city', 'state', 'latitude', 'longitude', 'stars', 'review_count', 'categories']]
    business_raw.to_csv(const.business_raw_file_name, sep='\t', index=False, header=False)

    logger.info('end process:' + sys._getframe().f_code.co_name)

    return business_raw


# Error in OSX in reading json file larger than 2GB due to possible bug in read_json. Works well in linux.
def review_json_to_rows():
    logger.info('start process:' + sys._getframe().f_code.co_name)

    review_raw = pd.read_json(const.review_json_file_name, lines=True)
    review_raw = review_raw[['review_id', 'user_id', 'business_id', 'stars', 'text']]
    review_raw.to_csv(const.review_raw_file_name, sep='\t', index=False, header=False)

    logger.info('end process:' + sys._getframe().f_code.co_name)

    return review_raw


def split_train_validate():
    logger.info('start process:' + sys._getframe().f_code.co_name)

    temp_review_raw = pd.read_csv(const.review_raw_file_name, sep='\t',
                                  names=['review_id', 'user_id', 'business_id', 'stars', 'text'], lineterminator='\n')

    temp_review_raw = temp_review_raw[['stars', 'text']]

    temp_review_raw['label'] = temp_review_raw.apply(
        lambda row: '__label__BAD' if str(row['stars']) == '1' else
        ('__label__BAD' if str(row['stars']) == '2' else ('__label__GOOD' if str(row['stars']) == '4' else
                                                          ('__label__GOOD' if str(row['stars']) == '5' else
                                                           '__label__NEUTRAL'))), axis=1)

    temp_review_raw['review'] = temp_review_raw.apply(
        lambda row: str(row['text']).lower().replace('\t', ' ').replace('\n', ' '), axis=1)

    temp_review_raw = temp_review_raw[['label', 'review']]

    msk = np.random.rand(len(temp_review_raw)) < 0.8
    train_raw = temp_review_raw[msk]
    validate_raw = temp_review_raw[~msk]

    train_raw.to_csv(const.train_raw_file_name, sep='\t', index=False, header=False)
    validate_raw.to_csv(const.validate_raw_file_name, sep='\t', index=False, header=False)

    logger.info('end process:' + sys._getframe().f_code.co_name)

    return train_raw, validate_raw


class StopWordDictionary(object):
    with open(const.stop_file_name) as f:
        stop_dict = f.readlines()

    stop_dictionary = [x.strip() for x in stop_dict]


def remove_stop_words_from_text(input_text):
    input_text_formatted = str(input_text).lower().replace('\"', '').replace("\t", " ")

    stop_words = map(re.escape, StopWordDictionary.stop_dictionary)
    output_text = re.sub(r"\b%s\b" % '\\b|\\b'.join(stop_words), ' ', input_text_formatted)

    return output_text


def remove_stop_words():
    logger.info('start process:' + sys._getframe().f_code.co_name)

    review_raw = pd.read_csv(const.review_raw_file_name, sep='\t',
                             names=['review_id', 'user_id', 'business_id', 'stars', 'text'], lineterminator='\n')
    train_raw = pd.read_csv(const.train_raw_file_name, sep='\t', names=['label', 'review'], lineterminator='\n')
    validate_raw = pd.read_csv(const.validate_raw_file_name, sep='\t', names=['label', 'review'], lineterminator='\n')

    review_raw['review'] = review_raw.apply(lambda row: remove_stop_words_from_text(row['text']), axis=1)
    train_raw['review'] = train_raw.apply(lambda row: remove_stop_words_from_text(row['review']), axis=1)
    validate_raw['review'] = validate_raw.apply(lambda row: remove_stop_words_from_text(row['review']), axis=1)

    review_raw.to_csv(const.review_processed_file_name, sep='\t', index=False, header=False)
    train_raw.to_csv(const.train_processed_file_name, sep='\t', index=False, header=False)
    validate_raw.to_csv(const.validate_processed_file_name, sep='\t', index=False, header=False)

    train_binary = train_raw.query("label in ('__label__GOOD', '__label__BAD') ")
    validate_binary = validate_raw.query("label in ('__label__GOOD', '__label__BAD') ")

    train_binary.to_csv(const.train_processed_binary_file_name, sep='\t', index=False, header=False)
    validate_binary.to_csv(const.validate_processed_binary_file_name, sep='\t', index=False, header=False)

    logger.info('end process:' + sys._getframe().f_code.co_name)


if __name__ == '__main__':

    business_json_to_rows()
    review_json_to_rows()
    split_train_validate()

    remove_stop_words()
