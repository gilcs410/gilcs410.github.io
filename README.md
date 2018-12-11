# CS410 Course project - Review sentiment analysis using Fasttext

## Objective

As an UIUC 'CS 410: Text Information Systems' course project, the objective is to generate a machine learning model 
that can determine the sentiment (positive vs negative) of given text. It uses 1~5 star-labelled business review data 
from Yelp open data as source. After certain pre-processing, data then gets fed into Fasttext (word2vec) module for 
training. After the model has been generated, the model is used for assigning sentiment score for the whole Yelp 
review open data as discriminative as possible (rather than just star-1, 2, 3, 4, 5) and re-ranks them in clustered 
groups based on different meta data like cuisine, city, etc. 

## Visualization

Please visit [here](https://gilcs410.github.io) for interactive visualization presenting the output of this project.
This page is generated using [Fullpage](https://github.com/alvarotrigo/fullpage.js) javascript with embedding 
[Tableau](https://www.tableau.com) dashboard.


## Data source

Yelp open data that has been used for this project can be downloaded [here](https://www.yelp.com/dataset).


## Model training steps

####1. Download Yelp open data set
This project uses Yelp data set challenge open data (Round 12). It contains 19,564,818 reviews 
on 188,593 business in US and Canada (However, this data set does not cover 100% of cities though). 
The file format is in JSON. Download it from [here](https://www.yelp.com/dataset). All we need is two files: 
`yelp_academic_dataset_business.json, yelp_academic_dataset_review.json`

####2. Pre-process the text data

There are several pre-processing step needed before training the model:
- Transform JSON format data into rows and columns using Pandas
- Convert all characters into lower case
- remove some stop keywords and special characters that might not be very helpful in determining the sentiment
- Label 'BAD' for reviews that has 1,2 stars 'GOOD' for 4, 5 stars.
- Randomly split 80% of the rows into train set. The rest of 20% will be used to check the performance of the model.

All you need to do is run the following command.
```bash
python3 preprocess.py
```

####3. Train the model using Fasttext (word2vec)
Fasttext is a library for efficient text classification and representation learning. It is similar as word2vec 
which embeds words into vector representative using word co occurrence information within certain window. 
The main difference between word2vec and Fasttext is that Fasttext can use sub-word ngram when embedding 
the words (e.g. Apple -> 'ap', 'app', 'appl', 'apple', 'ppl', 'pple', 'pple', 'ple', 'le'). For more information, 
please visit [here](https://fasttext.cc/).

After pre-processing job is finished, in order to proceed with the training, please run the following command.
```bash
python3 train.py
```

Using the pre-defined hyper parameter, the train process will iterate from 1 to 30 epoch. It will
automatically calculate precision, recall and F1 against the 20% validation set that was split previously in 
pre-processing. It will track the best performing version (in terms of micro F1), removing the other least performaning 
ones and saving only the best performing with the optimum epoch. It will produce logs to help
track the current status like below example.
```bash
epoch:1: micro precision:0.9569, micro_recall:0.9654, micro_f1:0.9612, macro_precision:0.9259, macro_recall:0.9184, macro_f1:0.9222, lr:0.01, duration:0:05:47 ====> Model improved!!!!
epoch:2: micro precision:0.9642, micro_recall:0.9703, micro_f1:0.9672, macro_precision:0.9373, macro_recall:0.9319, macro_f1:0.9346, lr:0.01, duration:0:07:31 ====> Model improved!!!!
epoch:3: micro precision:0.9677, micro_recall:0.9725, micro_f1:0.9701, macro_precision:0.9426, macro_recall:0.9383, macro_f1:0.9405, lr:0.01, duration:0:09:16 ====> Model improved!!!!
```

####4. Assign the sentiment score to all reviews

With the average sentiment model generated, each reviews in the Yelp open data set has been assigned with a sentiment 
probability between 0~1 (0: BAD, 1: GOOD) using predict_proba function provided in Fasttext.
Run below command, and you can get a file with all reviews along with their sentimental scores.

```bash
python3 assign.py [path_to_model_bin_file]
```

Check out the visualization [here](https://gilcs410.github.io) to see how this data can be used.

## Project presentation

[Download PPT](https://github.com/gilcs410/gilcs410.github.io/blob/master/cs410_presentation.pptx?raw=true)