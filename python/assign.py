# -*- coding: utf-8 -*-
"""

"""

__author__ = "Gil"
__status__ = "test"
__version__ = "0.1"
__date__ = "Nov 2018"

import const
from pyfasttext import FastText
import sys
import logging
import pandas as pd
pd.options.mode.chained_assignment = None

FORMAT = const.LOG_MSG_FORMAT
logging.basicConfig(format=FORMAT, datefmt=const.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_score(input_model, text):
    try:
        predict = input_model.predict_proba_single(str(text), k=1)[0]

        if predict[0] == 'GOOD':
            result = predict[1]
        else:
            result = 1 - predict[1]

        return result

    except IndexError:
        return 0.5


def assign_score_to_review(input_model):
    logger.info('start process:' + sys._getframe().f_code.co_name)

    review_processed = pd.read_csv(const.review_processed_file_name, sep='\t',
                                   names=['review_id', 'user_id', 'business_id', 'stars', 'text', 'review'],
                                   lineterminator='\n')

    review_processed['predict_score'] = review_processed.apply(lambda row: predict_score(input_model, row['review']), axis=1)
    review_processed.to_csv(const.review_processed_score_file_name, sep='\t', index=False, header=False)

    business_raw = pd.read_csv(const.business_raw_file_name, sep='\t',
                               names=['business_id', 'name', 'city', 'state', 'latitude', 'longitude', 'stars',
                                      'review_count', 'categories'], lineterminator='\n')

    review_processed['review_stars'] = review_processed['stars']
    review_processed_short = review_processed[['business_id', 'review_stars', 'predict_score']]

    business_score = business_raw.merge(review_processed_short, on=['business_id'])

    business_score_agg = business_score.groupby(
        ['business_id', 'name', 'city', 'state', 'latitude', 'longitude', 'stars', 'review_count', 'categories'],
        as_index=False).agg({'review_stars': 'mean', 'predict_score': 'mean'})
    business_score_agg.to_csv(const.business_score_file_name, sep='\t', index=False, header=False)

    logger.info('end process:' + sys._getframe().f_code.co_name)

    return business_score_agg


def assign_score_to_review_with_text():
    logger.info('start process:' + sys._getframe().f_code.co_name)

    business_raw = pd.read_csv(const.business_raw_file_name, sep='\t',
                               names=['business_id', 'name', 'city', 'state', 'latitude', 'longitude', 'stars',
                                      'review_count', 'categories'], lineterminator='\n')

    review_processed = pd.read_csv(const.review_processed_score_file_name, sep='\t',
                                   names=['review_id', 'user_id', 'business_id', 'stars', 'text', 'review',
                                          'predict_score'], lineterminator='\n')
    review_processed['review_stars'] = review_processed['stars']
    review_processed_short = review_processed[['business_id', 'review_stars', 'text', 'predict_score']]

    business_review_score = business_raw.merge(review_processed_short, on=['business_id'])
    business_review_score = business_review_score[
        ['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'categories', 'review_stars', 'text',
         'predict_score']]
    business_review_score_short = business_review_score.query(" stars in ('1', '5')")
    business_review_score_short.to_csv(const.business_review_score_file_name, sep='\t', index=False, header=False)

    logger.info('end process:' + sys._getframe().f_code.co_name)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        raise Exception('Need to run command as python assign.py [path_to_model_bin_file]')
    else:
        model_file = sys.argv[1]

    model = FastText()
    model.load_model(model_file)

    assign_score_to_review(model)
    assign_score_to_review_with_text()
