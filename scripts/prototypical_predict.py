import pandas as pd
import numpy as np
import nltk
import modin
import swifter
import collections
from importlib import reload
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
import logging.handlers
from prettytable import PrettyTable
from datetime import date
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tabulate import tabulate
pd.options.display.max_colwidth = 5000
# ---------------------------------------------


class Predict:
    def __init__(self, label):
        self.proto_cwp = pd.DataFrame()
        self.label = label
        self.test = pd.DataFrame()
        self.counter_t = None
        self.pred = pd.DataFrame()
        return

    def predict(self, data_path, proto_cwp_path):
        self.log.info("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
        self.log.info("PREDICTING DATA")
        self.log.info("Reading data")
        self.test = pd.read_feather(data_path)
        self.proto_cwp = pd.read_feather(proto_cwp_path)
        self.log.info("Starting prediction")
        self.log.info("A test data sample")
        self.log.info(tabulate(self.test.sample(3), headers='keys', tablefmt='psql'))

        long_str_t = ' '.join(self.test.X)
        tokens_t = long_str_t.split()
        self.WordCloudGen(long_str_t, 'Test data', f'../data/images/{self.label}_test.png')
        self.log.info(f"Total number of words is {len(tokens_t)}")
        self.counter_t = collections.Counter(tokens_t)
        self.log.info(f"Number of distinct words is {len(self.counter_t)}")
        table = PrettyTable()
        table.title = 'Most common 15 words'
        table.add_column(f'{self.label}', np.transpose(self.counter_t.most_common(15))[0])
        self.log.info(table)
        self.log.info(table.get_latex_string())
        self.GetPred()

        return

    def GetPred(self):
        self.log.info("Computing validation set predictions")
        self.pred = pd.DataFrame()
        self.log.info(
            "The class prob of a post is the sum of class|word probabilities for all proto word in class and in post")

        temp = self.proto_cwp[['sc_1', 'sc_0']]
        self.pred[['sc_1', 'sc_0']] = self.test.X.swifter.apply(
            lambda d: temp[self.proto_cwp.word.isin(d.split())].sum())[['sc_1', 'sc_0']]

        self.pred['Y'] = self.test.Y  # 1 for p and 0 for np

        self.pred['Y_pred'] = (self.pred.sc_1 > self.pred.sc_0)  # .astype(int)
        # pred['accuracy'] = (pred.Y == pred.Y_pred).astype(int)
        self.pred['nonresp_flag'] = 0
        self.pred.loc[self.pred.eval('sc_1 == sc_0 == 0'), 'nonresp_flag'] = 1
        self.log.info('saving to disk')
        # save to disk
        # pred.to_feather(f'../data/feather_files/{label}_pred_{k}.feather')
        self.log.info(f"saved as pred_{self.k}")
        self.log.info(tabulate(self.pred.sample(3), headers='keys', tablefmt='psql'))
        del self.proto_cwp, self.test
        return
