# -*- coding: utf-8 -*-
"""

"""

__author__ = "Gil"
__status__ = "test"
__version__ = "0.1"
__date__ = "Nov 2018"

LOG_MSG_FORMAT = '[%(levelname)s] %(asctime)s %(message)s'
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

business_json_file_name = 'data/yelp_academic_dataset_business.json'
business_raw_file_name = 'data/business_raw.txt'
business_score_file_name = 'data/business_score.txt'
business_review_score_file_name = 'data/business_review_score.txt'
business_review_score_word_file_name = 'data/business_review_score_word.txt'
review_json_file_name = 'data/yelp_academic_dataset_review.json'
review_raw_file_name = 'data/review_raw.txt'
review_processed_file_name = 'data/review_processed.txt'
review_processed_score_file_name = 'data/review_processed_score.txt'
train_raw_file_name = 'data/train_raw.txt'
train_processed_file_name = 'data/train_processed.txt'
train_processed_binary_file_name = 'data/train_processed_binary.txt'
validate_raw_file_name = 'data/validate_raw.txt'
validate_processed_file_name = 'data/validate_processed.txt'
validate_processed_binary_file_name = 'data/validate_processed_binary.txt'
stop_file_name = 'data/stop_words.txt'

JSON_KEY_ROOT = 'sentiment'
JSON_KEY_SUCCESS = "success"
JSON_KEY_ERROR = "error"