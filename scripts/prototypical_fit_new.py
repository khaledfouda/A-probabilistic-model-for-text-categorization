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


def SCORER(y_true, y_pred, flag):
    scr = pd.Series(dtype=np.float32)
    scr["nonresponse"] = flag.value_counts().loc[1] / len(flag)
    scr["Accuracy"] = accuracy_score(y_true, y_pred)
    scr["f1_score"] = f1_score(y_true, y_pred)
    scr['Precision'] = precision_score(y_true, y_pred)
    scr['Recall'] = recall_score(y_true, y_pred)
    scr["Accuracy_a"] = accuracy_score(y_true[flag == 0], y_pred[flag == 0])
    scr["f1_score_a"] = f1_score(y_true[flag == 0], y_pred[flag == 0])
    scr['Precision_a'] = precision_score(y_true[flag == 0], y_pred[flag == 0])
    scr['Recall_a'] = recall_score(y_true[flag == 0], y_pred[flag == 0])
    return scr


class Proto:
    def __init__(self, label, k, alpha=.5, harmonic_pscore=False, log_to_file=False):
        self.label = label.upper()
        self.k = k
        self.alpha = alpha
        self.harmonic_pscore=harmonic_pscore
        self.test, self.WP = pd.DataFrame(), pd.DataFrame()
        self.train, self.valid = pd.DataFrame(), pd.DataFrame()
        self.counter_t, self.counter_1, self.counter_0 = None, None, None
        self.log = None
        self.pred = pd.DataFrame()
        self.worddict, self.proto_cwp, self.proto = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.proto_train, self.proto_valid = pd.DataFrame(), pd.DataFrame()
        self.scores = pd.DataFrame()
        self.sum_proto = pd.Series()
        self.init_log(log_to_file)

        return

    def fit(self, data, valid_size=.2, n_drops=50):
        self.log.info("Starting fit")
        self.log.info(f"Hyperparameters: k={self.k}, alpha={self.alpha}, harmonic pscore={self.harmonic_pscore}")
        self.log.info("A data sample")
        self.log.info(tabulate(data.sample(3), headers='keys', tablefmt='psql'))
        self.log.info(f"The shape of the data is {data.shape}")
        self.log.info(data.Y.value_counts() / data.shape[0])
        # ---------------------
        # Splitting to train and test
        self.train, self.valid = train_test_split(data, test_size=valid_size, random_state=100,
                                                  shuffle=True, stratify=data.Y)
        del data
        self.train.reset_index(drop=True, inplace=True)
        self.valid.reset_index(drop=True, inplace=True)
        self.log.info(f"Taking {round(.2 * 100, 2)}% test subset.")
        self.log.info(f"The resulting train shape is {self.train.shape} and test shape is {self.valid.shape}")
        # ----------------------------------
        # transforming text
        # get text columns for training datasets
        self.GetCounters()
        # ----------------------------------------------------------------
        # Creating worddict
        self.WP_MODEL()
        # self.WordDict(n_drops)
        del self.counter_0, self.counter_1
        # creating wordclouds of the new worddict
        self.WordCloudGen(' '.join(self.WP.query("y1 == 1").word), 'Top of class 1',
                          f'../data/images/{self.label}_top_class1_k={self.k}.png')
        self.WordCloudGen(' '.join(self.WP.query("y0 == 0").word), 'Top of class 0',
                          f'../data/images/{self.label}_top_class0_k={self.k}.png')
        # -------------------------------------------------------------------
        self.train.Y_pred, self.train.nonresp_flag = self.predict(self.train.X)
        self.valid.Y_pred, self.valid.nonresp_flag = self.predict(self.valid.X)
        # ----------------------------------------------------------------------
        # Evaluating the model
        self.scores = pd.DataFrame()
        self.scores[f'Train_{self.k}'] = SCORER(self.train.Y, self.train.Y_pred,
                                                self.train.nonresp_flag)
        self.scores[f'Valid_{self.k}'] = SCORER(self.valid.Y, self.valid.Y_pred,
                                                self.valid.nonresp_flag)
        self.scores.reset_index().to_feather(f'../data/feather_files/{self.label}_scores_k={self.k}.feather')
        self.log.info(tabulate(self.scores, headers='keys', tablefmt='psql'))
        self.log.info(tabulate(self.scores, headers='keys', tablefmt='latex_raw'))
        # -------------------------------------------------------------------------------
        self.log.info("End")
        self.log.info("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    # ----------------------------------------

    def fit_CV(self, data, valid_size=.2, std=True, params={}):
        if std:
            params = {
                'alpha': [.5],
                'k': [200, 400, 600, 800, 1000, 1200, 1400]
            }
        best_score = -1
        best_k = -1
        best_alpha = -1
        # keys = params.keys()
        # values = (params[key] for key in keys)
        # combinations = [dict(zip(keys, combination)) for combination in params.product(*values)]
        # # --------------------------------
        self.log.info("A data sample")
        self.log.info(tabulate(data.sample(3), headers='keys', tablefmt='psql'))
        self.log.info(f"The shape of the data is {data.shape}")
        self.log.info(data.Y.value_counts() / data.shape[0])
        # ---------------------
        # Splitting to train and test
        self.train, self.valid = train_test_split(data, test_size=valid_size, random_state=100,
                                                  shuffle=True, stratify=data.Y)
        del data
        self.train.reset_index(drop=True, inplace=True)
        self.valid.reset_index(drop=True, inplace=True)
        self.log.info(f"Taking {round(.2 * 100, 2)}% test subset.")
        self.log.info(f"The resulting train shape is {self.train.shape} and test shape is {self.valid.shape}")
        # ----------------------------------
        # transforming text
        # get text columns for training datasets
        self.GetCounters()
        # ----------------------------------------------------------------
        for k in params['k']:
            self.k = k
            self.harmonic_pscore = True
            self.cvstep()
            if self.scores.loc['f1_score', 'valid'] > best_score:
                best_score = self.scores.loc['f1_score', 'valid']
                best_k = self.k
                best_alpha = -1
                self.log.info(
                    f"@@@ Better score achieved at  k={self.k}, harmonic pscore={self.harmonic_pscore}")
                self.log.info(f"current best score is {best_score}")
            self.harmonic_pscore = False
            for alpha in params['alpha']:
                self.alpha = alpha
                self.cvstep()
                if self.scores.loc['f1_score','valid'] > best_score:
                    best_score = self.scores.loc['f1_score','valid']
                    best_k = self.k
                    best_alpha = self.alpha
                    self.log.info(f"@@@ Better score achieved at  k={self.k}, alpha={self.alpha}, harmonic pscore={self.harmonic_pscore}")
                    self.log.info(f"current best score is {best_score}")

            self.log.info(f"@@@ reached end of training.")
            self.log.info(f"best score is {best_score}, and best parameters are:")
            self.log.info(f"k={best_k}, alpha={best_alpha}")
        return
    # -----------------------------------------------------------------------------------------

    def cvstep(self):
        # ----------------------------------------------------------------
        self.log.info("Starting CV fit step")
        self.log.info(f"Hyperparameters: k={self.k}, alpha={self.alpha}, harmonic pscore={self.harmonic_pscore}")
        # Creating worddict
        self.WP_MODEL()
        # self.WordDict(n_drops)
        # creating wordclouds of the new worddict
        self.WordCloudGen(' '.join(self.WP.query("y1 == 1").word), 'Top of class 1',
                          f'../data/images/{self.label}_top_class1_k={self.k}.png')
        self.WordCloudGen(' '.join(self.WP.query("y0 == 0").word), 'Top of class 0',
                          f'../data/images/{self.label}_top_class0_k={self.k}.png')
        # -------------------------------------------------------------------
        self.valid.Y_pred, self.valid.nonresp_flag = self.predict(self.valid.X)
        # ----------------------------------------------------------------------
        # Evaluating the model
        self.scores = pd.DataFrame()
        self.scores['valid'] = SCORER(self.valid.Y, self.valid.Y_pred,
                                                self.valid.nonresp_flag)
        self.log.info(tabulate(self.scores, headers='keys', tablefmt='psql'))
        self.log.info(tabulate(self.scores, headers='keys', tablefmt='latex_raw'))
        return

    def init_log(self, log_to_file=False):
        reload(logging)
        self.log = logging.getLogger("Bot")
        if self.log.hasHandlers():
            self.log.handlers.clear()
        logging.shutdown()
        if log_to_file:
            logging.basicConfig(filename=f"../data/log/{self.label}_log_k={self.k}__" +
                                         f"{date.today().strftime('%d-%m-%Y')}__.log",
                                filemode='a',
                                format='%(asctime)s: %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
            self.log = logging.getLogger("Bot")
            self.log.addHandler(logging.StreamHandler())
            self.log.info("######################################################################################")

        else:
            self.log = logging.getLogger("Bot")
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(logging.StreamHandler())
        return
# --------------------------------------------------

    def WordCloudGen(self, text, title, outfile):
        wordcloud = WordCloud(background_color="white", max_words=5000,
                              contour_width=6, contour_color='steelblue', scale=3)
        # Generate the first
        self.log.info("Word Cloud")
        self.log.info(title)
        wordcloud.generate(text)
        wordcloud.to_image()
        wordcloud.to_file(outfile)
        return
    # ------------------------------------------------------

    def GetCounters(self):
        self.log.info("dividing data into classes")
        y1 = self.train.query("Y == 1").X
        y0 = self.train.query("Y == 0").X
        y1_valid = self.valid.query("Y==1").X
        y0_valid = self.valid.query("Y==0").X
        # Get a long string of all text for each of the categories to analyse
        self.log.info("Joining the series of text into one string per category")
        long_str_1 = ' '.join(y1)
        long_str_0 = ' '.join(y0)
        long_str_1_v = ' '.join(y1_valid)
        long_str_0_v = ' '.join(y0_valid)
        # Transform that string into a list of strings
        self.log.info("Dividing those long strings into lists of words")
        self.tokens_1 = long_str_1.split()
        self.tokens_0 = long_str_0.split()
        tokens_1_v = long_str_1_v.split()
        tokens_0_v = long_str_0_v.split()
        # --------------------------------------------------
        # Generate wordclouds
        #self.WordCloudGen(long_str_1 + long_str_1_v, 'class 1', f'../data/images/{self.label}_class1_k={self.k}.png')
        #self.WordCloudGen(long_str_0 + long_str_0_v, 'class 0', f'../data/images/{self.label}_class0_k={self.k}.png')
        # ----------------------------------------------
        # Counting the occurrences of each of the words. Output: list (word, #occurences)
        self.log.info("Counting the occurrences of each word per class")
        self.log.info(f"Total umber of words (training+validation) is:")
        self.log.info(f"Class 1: {len(self.tokens_1) + len(tokens_1_v)}, Class 0: {len(self.tokens_0) + len(tokens_0_v)}")
        self.counter_1 = collections.Counter(self.tokens_1)
        self.counter_0 = collections.Counter(self.tokens_0)
        self.log.info(f"Number of distinct training words for each class is")
        self.log.info(f"{len(self.counter_1)} and {len(self.counter_0)}")
        self.log.info("Visualizing the top 15 common words in each category [latex code below]")
        table = PrettyTable()
        table.title = 'Most common 15 words in each category '
        table.add_column('class 1', np.transpose(collections.Counter(self.tokens_1 + tokens_1_v).most_common(15))[0])
        table.add_column('class 0', np.transpose(collections.Counter(self.tokens_0 + tokens_0_v).most_common(15))[0])
        self.log.info(table)
        self.log.info(table.get_latex_string())
        return
    # -----------------------------------------------------------------------------

    def WP_MODEL(self):
        self.log.info("*_* Inside GetWP()")
        # 1. P(w|y)
        # use x1=long_str_p, x0=long_str_np,
        w = pd.Series(list(set(self.tokens_0) | set(self.tokens_1)))
        pwy = pd.DataFrame(data=0, index=w, columns=[0, 1])
        pwy.loc[self.counter_0.keys(), 0] = list(self.counter_0.values())
        pwy.loc[self.counter_1.keys(), 1] = list(self.counter_1.values())
        self.pywp = pwy.copy()
        pwy[0] = pwy[0] / len(self.tokens_0)
        pwy[1] = pwy[1] / len(self.tokens_1)

        # 2. P(w)
        t = self.train.X.map(lambda d: " ".join(set(d.split())))
        t = ' '.join(t).split()
        pw = pd.Series(collections.Counter(t))
        pw = pw.divide(len(self.train))
        # 3. PScore
        if self.harmonic_pscore:
            pscore = (2 * pwy.multiply(pw, axis=0)) / pwy.add(pw, axis=0)
        else:
            pscore = (self.alpha * pwy).add((1-self.alpha) * pw, axis=0)
        #
        # 4. choose top k
        t0 = pscore[0].sort_values(ascending=False)[0:self.k].index
        t1 = pscore[1].sort_values(ascending=False)[0:self.k].index
        self.WP = pd.DataFrame({'word': list(set(t0) | set(t1)), 'y0': 0, 'y1': 0})
        self.WP.index = self.WP.word
        self.WP.loc[t0, 'y0'] = 1
        self.WP.loc[t1, 'y1'] = 1
        self.log.info(f"WP created. Number of WP words are {len(self.WP)}")
        # 5.  computing p(y|wp)
        self.pywp = self.pywp.loc[self.WP.index, ]
        psum = self.pywp[0] + self.pywp[1]
        self.pywp = self.pywp.divide(psum, axis=0)
        # self.pywp.reset_index().to_feather(f"{self.label}_pywp_{self.k}.feather")
        self.log.info(f"P(y|wp) is calculated and saved to disk as {self.label}_pywp_{self.k}.feather")

        return
    # --------------------------------------------------------------------------------

    def predict(self, x=pd.Series()):
        pred = x.swifter.allow_dask_on_strings()\
            .apply(lambda d: self.pywp.loc[self.pywp.index.intersection(d.split()), :].sum(axis=0).argmax())
        flag = x.swifter.apply(lambda d: len(self.pywp.index.intersection(d.split())) == 0)
        return pred, flag
    # --------------------------------------------------------------------------------

    def WordDict(self, n_drop):
        self.log.info("Creating a table of the number of occurrences of each word in each of the two classes.")
        # Make a table of word - #occurances in class 1 - #occurances in class 2 [saving to file]
        self.worddict = pd.DataFrame({'word': self.counter_1.keys(), 'c1_occ': self.counter_1.values()})
        self.worddict['c0_occ'] = self.worddict.word.swifter.apply(lambda x: self.counter_0.get(x, 0))
        # ----------------------------- class 2
        temp = pd.DataFrame({'word': self.counter_0.keys(), 'c0_occ': self.counter_0.values()})
        temp['c1_occ'] = temp.word.swifter.apply(lambda x: self.counter_1.get(x, 0))
        # Merge
        self.worddict = pd.concat((self.worddict, temp))\
            .drop_duplicates('word').sort_values(by='c1_occ', ascending=False)
        # ------------------------------------------------------------------
        # log
        self.log.info(f"Number of distinct words is {len(self.worddict)}")
        self.log.info(f"Dropping words occurring less than {n_drop} times")
        # Drop words occurring less than n times
        self.worddict = self.worddict.query('c1_occ >= @n_drop or c0_occ >= @n_drop').reset_index(drop=True)
        self.log.info(f" number of words occurring at least {n_drop} times is {len(self.worddict)} words")

        # Computer score a
        self.log.info("Computing proto score. Equation 1. Objective, choose top k words")
        self.worddict['sc_sum'] = self.worddict.c1_occ + self.worddict.c0_occ
        self.worddict['sc_1'] = self.worddict.c1_occ / self.worddict.sc_sum
        self.worddict['sc_0'] = self.worddict.c0_occ / self.worddict.sc_sum
        # Remove non-words (ie. names)
        self.log.info("Removing names and other non-recognizable words.")
        self.log.info("Unfortunately, some names would be detected since they hold a second meaning")
        self.worddict = self.worddict[self.worddict.word.isin(nltk.corpus.words.words('en'))]
        # Keep top k words in each class -
        # k = 400
        self.log.info(f"keeping top {self.k} words in each class")
        temp = self.worddict.sort_values(by='sc_1', ascending=False)[0:self.k]
        temp['Y'] = 1
        self.worddict = self.worddict.sort_values(by='sc_0', ascending=False)[0:self.k]
        self.worddict['Y'] = 0
        self.worddict = pd.concat((temp, self.worddict))
        self.worddict = self.worddict.drop_duplicates('word') \
            .sort_values(by='sc_1', ascending=False).reset_index(drop=True)
        self.log.info(f"Number of words after keeping top {self.k} words is {len(self.worddict)}")
        # save to disk
        # worddict.to_feather(f'../data/feather_files/{label}_worddict_k={k}.feather')
        return
    # -----------------------------------------------------------

    def Proto(self):
        # wp_in_u includes the frequency of the selected words in each of the posts
        # the shape is list of lists. posts x words
        self.log.info("Counting the occurrence of the chosen words inside each of the posts.")
        self.log.info("The resulting dataframe is of shape (number of posts)x(2k).")
        wp_in_u = self.train.X.swifter.apply(lambda d: ([d.count(wp) for wp in self.worddict.word]))
        wp_in_u = pd.DataFrame(wp_in_u)
        self.log.info(f"saved as wp_in_u_{self.k} with the dimension of {wp_in_u.shape}")
        self.log.info(f"Transforming the previous variable into a dataframe. saved as wp_proto_{self.k}")
        self.proto = wp_in_u.swifter.apply(lambda d: pd.Series(d[0]), axis=1)
        self.proto.columns = self.worddict.word
        self.proto.index = self.train.index
        # sum of all words in each post
        self.log.info("Computing the sum of words in each post")
        self.sum_proto = self.train.X.swifter.apply(lambda d: len(d.split()))
        self.log.info("Creating the first set of features, equation 2.")
        self.log.info("(#occurence of wp in u)/(#words in u). A table of (#posts)x(2k)")
        self.proto = self.proto.divide(self.sum_proto, axis=0)
        # proto.to_feather(f'../data/feather_files/{label}_proto_{k}.feather')
        self.log.info(f"saved as proto_{self.k}")
        return
    # ---------------------------------------------------------

    def ProtoTrain(self):
        self.log.info("The next feature is a score per (post,class)")
        self.proto_train = pd.DataFrame()
        # The numerator
        self.proto_train['sc_1'] = self.proto[self.worddict.query("Y == 1").word].sum(axis=1) /\
            len(self.worddict.query("Y == 1"))
        self.proto_train['sc_0'] = self.proto[self.worddict.query("Y == 0").word].sum(axis=1) /\
            len(self.worddict.query("Y == 0"))
        # Divide by the denominator (same as the previous score)
        self.proto_train = self.proto_train.divide(self.sum_proto, axis=0)
        self.proto_train['Y'] = self.train.Y
        self.proto_train['Y_pred'] = (self.proto_train.sc_1 > self.proto_train.sc_0) 
        self.proto_train['nonresp_flag'] = 0
        self.proto_train.loc[self.proto_train.eval('sc_1 == sc_0 == 0'), 'nonresp_flag'] = 1
        self.log.info('saving to disk')
        # save to disk
        # proto_train.to_feather(f'../data/feather_files/{label}_proto_train_{k}.feather')
        # self.log.info(f"saved as proto_train_{self.k}")
        self.log.info(tabulate(self.proto_train.sample(3), headers='keys', tablefmt='psql'))
        del self.worddict, self.sum_proto, self.train
        return
    # ---------------------------------------------------------

    def ProtoCWP(self):
        self.log.info("Computing the probabilities of classes given proto words")
        self.log.info("The Y is an assignment to the class of higher probability")
        # probabilities of class given a proto word.
        # the Y is an  assignment to the class of higher probability
        # Rows are words
        self.proto_cwp = pd.DataFrame()
        self.proto_cwp['sc_1'] = self.proto.swifter.apply(lambda d: d.mul(self.proto_train.sc_1).sum())
        self.proto_cwp['sc_0'] = self.proto.swifter.apply(lambda d: d.mul(self.proto_train.sc_0).sum())
        score_sum = self.proto_cwp.sc_1 + self.proto_cwp.sc_0
        self.proto_cwp.sc_1 = self.proto_cwp.sc_1 / score_sum
        self.proto_cwp.sc_0 = self.proto_cwp.sc_0 / score_sum
        self.proto_cwp['Y'] = self.proto_cwp.swifter.apply(lambda d: d.sc_1 > d.sc_0, axis=1)
        self.proto_cwp['word'] = self.proto.columns
        self.log.info("dataframe was created successfully. Saving to disk...")
        self.log.info(tabulate(self.proto_cwp.sample(3), headers='keys', tablefmt='psql'))
        self.proto_cwp.reset_index(drop=True)\
            .to_feather(f'../data/feather_files/{self.label}_proto_CWP_k={self.k}.feather')
        self.log.info('saved to disk')
        del self.proto
        return
    # ---------------------------------------------------------

    def ProtoValid(self):
        self.log.info("Computing validation set predictions")
        self.proto_valid = pd.DataFrame()
        self.log.info(
            "The class prob of a post is the sum of class|word probabilities for all proto word in class and in post")

        temp = self.proto_cwp[['sc_1', 'sc_0']]
        self.proto_valid[['sc_1', 'sc_0']] = self.valid.X.swifter.apply(
            lambda d: temp[self.proto_cwp.word.isin(d.split())].sum())[['sc_1', 'sc_0']]

        self.proto_valid['Y'] = self.valid.Y  # 1 for p and 0 for np

        self.proto_valid['Y_pred'] = (self.proto_valid.sc_1 > self.proto_valid.sc_0)  
        self.proto_valid['nonresp_flag'] = 0
        self.proto_valid.loc[self.proto_valid.eval('sc_1 == sc_0 == 0'), 'nonresp_flag'] = 1
        self.log.info(tabulate(self.proto_valid.sample(3), headers='keys', tablefmt='psql'))
        del self.proto_cwp, self.valid
        return
    # ---------------------------------------------------------------------

    # -------------------------------------------------------------
    def oopredict(self, data_path):
        self.log.info("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
        self.log.info("PREDICTING DATA")
        self.log.info("Reading data")
        self.test = pd.read_feather(data_path)
        # self.proto_cwp = pd.read_feather(proto_cwp_path)
        self.proto_cwp = pd.read_feather(f'../data/feather_files/{self.label}_proto_CWP_k={self.k}.feather')
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

        self.pred['Y_pred'] = (self.pred.sc_1 > self.pred.sc_0)  
        
        self.pred['nonresp_flag'] = 0
        self.pred.loc[self.pred.eval('sc_1 == sc_0 == 0'), 'nonresp_flag'] = 1
        self.log.info('saving to disk')
        # save to disk
        # pred.to_feather(f'../data/feather_files/{label}_pred_{k}.feather')
        self.log.info(f"saved as pred_{self.k}")
        self.log.info(tabulate(self.pred.sample(3), headers='keys', tablefmt='psql'))
        del self.proto_cwp, self.test
        return


if __name__ == "__main__":
    pass
