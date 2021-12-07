import pandas as pd
import numpy as np
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
import time
pd.options.display.max_colwidth = 5000  # to accommodate long strings
# ---------------------------------------------


def SCORER(y_true, y_pred, flag):
    """
    Input:
        y_true: list-like or series of true response values.
        y_pred: list-like or series of predicted response values.
        flag: list-like or series of binary values.
    Description:
        SCORER is general-purpose function calculating the accuracy, precision, recall and F-score
        In addition, it would calculate the percentage of True values in the flag input, as well as,
        the previous scores from the data corresponding to the negative rows of flag.
    Return:
        dataframe
    """
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
        # setting initial values to all shared variables.
        self.label = label.upper()
        self.k = k
        self.alpha = alpha
        self.harmonic_pscore = harmonic_pscore
        # set place-holders.
        self.train, self.valid, self.scores, self.log, self.tokens_0, self.tokens_1,\
            self.counter_0, self.counter_1, self.WP, self.p_y_wp =\
            None, None, None, None, None, None, None, None, None, None
        # initiate log object
        self.init_log(log_to_file)
        return

    def fit(self, data, valid_size=.2):
        self.log.info("Starting fit")
        self.log.info(f"Hyperparameters: k={self.k}, alpha={self.alpha}, harmonic pscore={self.harmonic_pscore}")
        self.log.info("A data sample")
        self.log.info(tabulate(data.sample(3), headers='keys', tablefmt='psql'))
        self.log.info(tabulate(data.sample(3), headers='keys', tablefmt='latex_raw'))
        self.log.info(f"The shape of the data is {data.shape}")
        self.log.info(data.Y.value_counts() / data.shape[0])
        # ---------------------
        # Splitting to train and test
        self.train, self.valid = train_test_split(data, test_size=valid_size, random_state=100,
                                                  shuffle=True, stratify=data.Y)
        del data
        self.train.reset_index(drop=True, inplace=True)  # resetting index prevents errors in some methods
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
        self.WordCloudGen(' '.join(self.WP.query("y0 == 1").word), 'Top of class 0',
                          f'../data/images/{self.label}_top_class0_k={self.k}.png')
        # -------------------------------------------------------------------
        self.log.info("End")
        self.log.info("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    # ----------------------------------------

    def fit_CV(self, data, valid_size=.2, std=True, params=None):
        """
        This method applied cross-validation to select the optimal k and alpha.
        There is a standard parameters list provided, however, a user can include their own list.
        The input params must best a dictionary of alpha and k. The harmonic alternative is always considered.
        best_alpha = -1 if the harmonic is the optimal choice.
        """
        if std:
            params = {
                'alpha': [.5],
                'k': [200, 400, 600, 800, 1000, 1200, 1400]
            }
        best_score = -1
        best_k = -1
        best_alpha = -1
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
            self.cv_step()
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
                self.cv_step()
                if self.scores.loc['f1_score', 'valid'] > best_score:
                    best_score = self.scores.loc['f1_score', 'valid']
                    best_k = self.k
                    best_alpha = self.alpha
                    self.log.info(f"@@@ Better score achieved at  k={self.k}, alpha={self.alpha}, ")
                    self.log.info(f"harmonic pscore={self.harmonic_pscore}. current best score is {best_score}")

            self.log.info(f"@@@ reached end of training.")
            self.log.info(f"best score is {best_score}, and best parameters are:")
            self.log.info(f"k={best_k}, alpha={best_alpha}")
        return
    # -----------------------------------------------------------------------------------------

    def cv_step(self):
        """
        A helper function for fit_cv().
        Applies a fitting and valid prediction step.
        """
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
    # -----------------------------------------------------------------------------

    def init_log(self, log_to_file=False):
        """
        Initialize the logging object.
        There are two option for logging, either to log both to screen and file
        or to log only to screen
        """
        # The following 5 lines is to shutdown any leftover handlers from previous runs.
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
        else:
            self.log = logging.getLogger("Bot")
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(logging.StreamHandler())
        # test run
        self.log.info("********** START OF LOG ****************")
        return
    # --------------------------------------------------

    def WordCloudGen(self, text, title, outfile):
        """
        Input:
            text: a string to produce a wordcloud to.
            title: The title of the produced image. (title is written to the logfile only)
            outfile: The name and path to the output image file. Expecting a png extension.
        Description:
            A wordcloud of text is produced and the figure is saved to the outfile.
            If ran from graphics-supporting console, the image will show at creation.
        Output:
            None
        """
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
        """
        The following prepares the data for computing the probabilities.
        It defines 4 new shared variables:
            tokens_0 and tokens_1: contain lists of all the words in each class.
            counter_1 and counter_1: Frequency dictionaries of tokens_0 and tokens_1
        It also provides more data summary and a wordcloud for each class.
        """
        self.log.info("dividing data into classes")
        y1 = self.train.query("Y == 1").X
        y0 = self.train.query("Y == 0").X
        y1_valid = self.valid.query("Y==1").X
        y0_valid = self.valid.query("Y==0").X
        # Get a long string of all text for each of the categories to analyse
        long_str_1 = ' '.join(y1)
        long_str_0 = ' '.join(y0)
        long_str_1_v = ' '.join(y1_valid)
        long_str_0_v = ' '.join(y0_valid)
        # Transform that string into a list of strings
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
        self.log.info(f"Class 1: {len(self.tokens_1) + len(tokens_1_v)}," +
                      f"Class 0: {len(self.tokens_0) + len(tokens_0_v)}")
        self.counter_1 = collections.Counter(self.tokens_1)
        self.counter_0 = collections.Counter(self.tokens_0)
        self.log.info(f"Number of distinct training words for each class is")
        self.log.info(f"{len(self.counter_1)} and {len(self.counter_0)}")
        # The following part prints the most common 15 words in each class.
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
        """
        This function computes all the probabilities defined for the model.
        """
        self.log.info("*_* Inside GetWP()")
        # 1. P(w|y) = (<w,X(y)) / ||X(y)||
        # use x1=long_str_p, x0=long_str_np,
        w = pd.Series(list(set(self.tokens_0) | set(self.tokens_1)))
        pwy = pd.DataFrame(data=0, index=w, columns=[0, 1])
        pwy.loc[self.counter_0.keys(), 0] = list(self.counter_0.values())
        pwy.loc[self.counter_1.keys(), 1] = list(self.counter_1.values())
        self.p_y_wp = pwy.copy()  # reuse for P(Y|wp)
        pwy[0] = pwy[0] / len(self.tokens_0)
        pwy[1] = pwy[1] / len(self.tokens_1)

        # 2. P(w) = #x containing w / length(x)
        t = self.train.X.map(lambda d: " ".join(set(d.split())))
        t = ' '.join(t).split()
        pw = pd.Series(collections.Counter(t))
        pw = pw.divide(len(self.train))
        # 3. PScore is either a harmonic mean or an alpha mean.
        # The harmonic is the default and it proved to give better results.
        # The alpha mean is provided for cross-validation purposes.
        if self.harmonic_pscore:
            pscore = (2 * pwy.multiply(pw, axis=0)) / pwy.add(pw, axis=0)
        else:
            pscore = (self.alpha * pwy).add((1-self.alpha) * pw, axis=0)
        #
        # 4. choose top k as wp
        t0 = pscore[0].sort_values(ascending=False)[0:self.k].index
        t1 = pscore[1].sort_values(ascending=False)[0:self.k].index
        self.WP = pd.DataFrame({'word': list(set(t0) | set(t1)), 'y0': 0, 'y1': 0})
        self.WP.index = self.WP.word
        self.WP.loc[t0, 'y0'] = 1
        self.WP.loc[t1, 'y1'] = 1
        self.log.info(f"WP created. Number of WP words are {len(self.WP)}")
        self.log.info(tabulate(self.WP.head(15), headers='keys', tablefmt='psql', showindex=False))
        self.log.info(tabulate(self.WP.head(15), headers='keys', tablefmt='latex_raw', showindex=False))
        # 5.  computing p(y|wp)
        self.p_y_wp = self.p_y_wp.loc[self.WP.index, ]
        sum_p = self.p_y_wp[0] + self.p_y_wp[1]
        self.p_y_wp = self.p_y_wp.divide(sum_p, axis=0)
        temp = pd.concat((self.p_y_wp.sort_values(by=1, axis=0, ascending=False).iloc[:7, :],
                          self.p_y_wp.sort_values(by=0, axis=0, ascending=False).iloc[:7, :]))
        self.log.info(tabulate(self.p_y_wp.sample(20), headers='keys', tablefmt='psql', showindex=True))
        self.log.info(tabulate(self.p_y_wp.sample(20), headers='keys', tablefmt='latex_raw', showindex=True))
        return
    # --------------------------------------------------------------------------------

    def train_valid_predict(self):
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
        return
    # ------------------------------------------------------

    def predict(self, x=pd.Series()):
        """
        Input:
            x: a series. (list-like objects are not accepted because of the use of pandas features)
        Require:
            The dataframe self.p_y_wp (ie, p(y|wp)) must be defined with legitimate values.
        Return:
            pred: A series of predictions (binary)
            flag: A series of non-response flag (binary)
        """
        pred = x.swifter.allow_dask_on_strings()\
            .apply(lambda d: self.p_y_wp.loc[self.p_y_wp.index.intersection(d.split()), :].sum(axis=0).argmax())
        flag = x.swifter.apply(lambda d: len(self.p_y_wp.index.intersection(d.split())) == 0)
        return pred, flag
    # --------------------------------------------------------------------------------

    def test_predict(self, x=pd.Series()):
        """
        This function is similar to predict() with the addition of producing word-clouds and
        prediction sample.
        """
        pred, flag = self.predict(x)
        nonresp = flag.value_counts().loc[1] / len(flag)
        self.log.info(f"Percentage of non-responses is {round(nonresp*100,2)}%")
        pred_t = pd.Series(pred).map(lambda d: 'political' if d else 'non-political')
        test_df = pd.DataFrame({'text': x, 'class': pred_t})
        self.log.info(tabulate(test_df.sample(15), headers='keys', tablefmt='psql'))
        self.log.info(tabulate(test_df.sample(15), headers='keys', tablefmt='latex_raw'))
        self.WordCloudGen(' '.join(test_df[pred == 1].text), 'Predicted class 1',
                          f'../data/images/{self.label}_test_class1_k={self.k}.png')
        self.WordCloudGen(' '.join(test_df[pred == 0].text), 'Predicted class 0',
                          f'../data/images/{self.label}_test_class0_k={self.k}.png')
        return pred, flag
    # -----------------------------------------------------------


if __name__ == "__main__":
    pass
